#!/usr/bin/env bash
# End-to-end smoke test: vLLM directly, then the backend in front of it.
# Re-runnable after a reboot once both services are back up.
#
#   ./scripts/smoke_test.sh
#
# Reads backend/.env for the key/endpoint/model so there is one source of truth.
# Exits non-zero on the first failed check.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$HERE/backend/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE" >&2; exit 1
fi

get() { grep "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2-; }

VLLM_URL="$(get HF_ENDPOINT)"
VLLM_KEY="$(get HF_TOKEN)"
MODEL="$(get HF_MODEL)"
API_KEY="$(get API_KEY)"
BACKEND="${BACKEND:-http://127.0.0.1:8010}"
WEB="${WEB:-http://127.0.0.1:8080}"
SESSION="smoke-$(date +%s)"

# Web-tier config lives in its own env file.
WEB_ENV="$HERE/web/.env"
webget() { grep "^$1=" "$WEB_ENV" 2>/dev/null | head -1 | cut -d= -f2-; }
ACCESS_CODE="$(webget ACCESS_CODE)"

pass=0; fail=0
ok()   { echo "  PASS  $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $1"; fail=$((fail+1)); }
hdr()  { echo; echo "=== $1 ==="; }

# Only send x-api-key when one is actually configured.
auth=()
[[ -n "$API_KEY" ]] && auth=(-H "x-api-key: $API_KEY")

# ---------------------------------------------------------------- vLLM ----
hdr "1. vLLM /v1/models reports exactly '$MODEL'"
got=$(curl -s -H "Authorization: Bearer $VLLM_KEY" "$VLLM_URL/v1/models" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
echo "  id = ${got:-<none>}"
[[ "$got" == "$MODEL" ]] && ok "model id matches" || bad "expected '$MODEL', got '${got:-<none>}'"

hdr "2. vLLM non-streaming chat completion"
body=$(curl -s -H "Authorization: Bearer $VLLM_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is sundowning in dementia? Answer in two sentences.\"}],\"max_tokens\":120,\"temperature\":0.7}" \
  "$VLLM_URL/v1/chat/completions")
text=$(echo "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
echo "  ${text:0:300}"
[[ ${#text} -gt 20 ]] && ok "coherent text returned" || bad "no usable text: $body"

hdr "3. vLLM streaming arrives incrementally"
tmp=$(mktemp)
curl -s -N -H "Authorization: Bearer $VLLM_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count slowly from 1 to 15.\"}],\"max_tokens\":120,\"stream\":true}" \
  "$VLLM_URL/v1/chat/completions" \
  | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > "$tmp"
first=$(grep -m1 'data: {' "$tmp" | cut -d' ' -f1)
last=$(grep 'data: {' "$tmp" | tail -1 | cut -d' ' -f1)
n=$(grep -c 'data: {' "$tmp")
spread=$(python3 -c "print(f'{float('$last')-float('$first'):.2f}')" 2>/dev/null || echo 0)
echo "  $n chunks, first-to-last spread ${spread}s"
python3 -c "import sys; sys.exit(0 if float('$spread') > 0.05 and $n > 5 else 1)" \
  && ok "chunks spread over time (not buffered)" || bad "chunks arrived all at once"
rm -f "$tmp"

hdr "4. vLLM 404s on a wrong model name"
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $VLLM_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"not-a-real-model","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
  "$VLLM_URL/v1/chat/completions")
echo "  HTTP $code"
[[ "$code" == "404" ]] && ok "wrong model name rejected" || bad "expected 404, got $code"

# ------------------------------------------------------------- backend ----
hdr "5. backend /togethercareAI"
h=$(curl -s "$BACKEND/togethercareAI"); echo "  $h"
want="$( [[ "$(get RAG_ENABLED)" == "1" ]] && echo true || echo false )"
echo "$h" | grep -q "\"rag_enabled\":$want" && ok "rag_enabled $want (matches .env)" || bad "rag_enabled != $want"

hdr "6. backend /togethercareAI/llm does a real generation"
h=$(curl -s "${auth[@]}" "$BACKEND/togethercareAI/llm"); echo "  ${h:0:300}"
echo "$h" | grep -q '"ok":true' && ok "llm reachable through backend" || bad "$h"

hdr "7. backend /generate (Sources expected only in RAG mode)"
a=$(curl -s "${auth[@]}" -H 'Content-Type: application/json' -H "sesh_id: $SESSION" \
  -d '{"message":"My mother has Alzheimers and gets agitated in the evening. What can I do?","max_new_tokens":160}' \
  "$BACKEND/generate")
echo "  ${a:0:400}"
echo "$a" | grep -q '"answer"' && ok "answer returned" || bad "$a"
# Sources are expected in RAG mode and must NOT appear in LLM-only mode.
RAG="$(get RAG_ENABLED)"
if [[ "$RAG" == "1" ]]; then
  if echo "$a" | grep -q 'Sources:'; then
    ok "Sources section present (RAG on)"
  else
    echo "  NOTE  no Sources for this query - expected when every chunk scores"
    echo "        below RAG_MIN_SCORE=$(get RAG_MIN_SCORE). Not a failure."
  fi
else
  echo "$a" | grep -q 'Sources:' && bad "Sources section present with RAG off" || ok "no Sources section"
fi

hdr "8. backend /generate_stream"
tmp=$(mktemp)
curl -s -N "${auth[@]}" -H 'Content-Type: application/json' -H "sesh_id: $SESSION" \
  -d '{"message":"Name three tips for a dementia caregiver.","max_new_tokens":160}' \
  "$BACKEND/generate_stream" > "$tmp"
echo "  $(grep -c '^data: ' "$tmp") SSE frames"
grep -q '"event": "meta"' "$tmp" && ok "meta frame present" || bad "no meta frame"
tail -2 "$tmp" | grep -q '\[DONE\]' && ok "[DONE] frame present" || bad "no [DONE] frame"
rm -f "$tmp"

hdr "9. backend /generate_ab returns two answers"
t0=$(date +%s.%N)
ab=$(curl -s "${auth[@]}" -H 'Content-Type: application/json' -H "sesh_id: $SESSION" \
  -d '{"message":"How do I talk to a parent who keeps repeating the same question?","max_new_tokens":160}' \
  "$BACKEND/generate_ab")
t1=$(date +%s.%N)
echo "  took $(python3 -c "print(f'{float('$t1')-float('$t0'):.1f}s')")"
echo "$ab" | grep -q '"answer_a"' && echo "$ab" | grep -q '"answer_b"' \
  && ok "two answers returned" || bad "$ab"

hdr "10. JSONL log line landed"
logf="$HERE/backend/logs/$SESSION.jsonl"
if [[ -f "$logf" ]]; then
  echo "  $(wc -l < "$logf") lines in backend/logs/$SESSION.jsonl"
  ok "log written"
else
  bad "no log at $logf"
fi

# ------------------------------------------------------------- web tier ----
# The public entrypoint: serves the built React app and proxies /api behind a
# shared access code. This is what testers actually hit.
cj="$(mktemp)"

hdr "11. web tier is up and reports its config"
w=$(curl -s --max-time 5 "$WEB/togethercareAI/web"); echo "  ${w:0:200}"
echo "$w" | grep -q '"service":"togethercareAI-web"' \
  && ok "web tier answering" || bad "web tier not answering on $WEB"

hdr "12. the gate actually gates"
# Nothing at all should be served before the code is entered - not the app
# shell, not the JS bundle.
echo "$(curl -s --max-time 5 "$WEB/")" | grep -q "Access code" \
  && ok "unauthenticated visitors get the gate, not the app" \
  || bad "gate page not served at /"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$WEB/api/togethercareAI")
[[ "$code" == "401" ]] && ok "/api refused without the code (HTTP 401)" \
  || bad "expected 401 from ungated /api, got $code"

hdr "13. the access code opens it"
if [[ -z "$ACCESS_CODE" ]]; then
  bad "no ACCESS_CODE in web/.env - cannot test the gate"
else
  curl -s -c "$cj" --max-time 5 -X POST -H 'Content-Type: application/json' \
    -d "{\"code\":\"$ACCESS_CODE\"}" "$WEB/api/auth" | grep -q '"ok":true' \
    && ok "correct code accepted" || bad "correct code rejected"
  curl -s --max-time 5 -X POST -H 'Content-Type: application/json' \
    -d '{"code":"definitely-not-it"}' "$WEB/api/auth" | grep -q '"error"' \
    && ok "wrong code rejected" || bad "wrong code was NOT rejected"
fi

hdr "14. backend API_KEY never reaches the browser"
# The single most important property of this deployment: the key protecting a
# shared university GPU must not be in anything served to a visitor.
leak=0
grep -rl "$API_KEY" "$HERE/frontend/dist/" >/dev/null 2>&1 && leak=1
[[ -n "$API_KEY" ]] && curl -s -b "$cj" --max-time 5 "$WEB/env-config.js" | grep -q "$API_KEY" && leak=1
[[ "$leak" == "0" ]] && ok "API_KEY absent from dist/ and from env-config.js" \
  || bad "API_KEY IS BEING SERVED TO THE BROWSER"

hdr "15. proxied /api reaches the backend with the key attached"
# No x-api-key is sent here on purpose - the web tier must add it.
w=$(curl -s -b "$cj" --max-time 30 "$WEB/api/togethercareAI/llm")
echo "  ${w:0:160}"
echo "$w" | grep -q '"ok":true' \
  && ok "authenticated backend route works through the proxy" || bad "$w"

hdr "16. SSE still streams through the proxy (not buffered)"
spread=$(curl -s -N -b "$cj" --max-time 90 -H 'Content-Type: application/json' \
  -H "sesh_id: $SESSION-web" \
  -d '{"message":"Name two ways to handle sundowning.","max_new_tokens":120}' \
  "$WEB/api/generate_stream" | python3 -c "
import sys, time
t0=None; last=0; n=0
for line in sys.stdin:
    if line.startswith('data: '):
        now=time.time()
        if t0 is None: t0=now
        last=now; n+=1
print(f'{n} {last-t0:.2f}')
")
echo "  $spread frames/seconds"
python3 -c "
import sys
n, dt = '$spread'.split()
sys.exit(0 if int(n) > 5 and float(dt) > 0.5 else 1)
" && ok "tokens arrived incrementally through the proxy" \
  || bad "streaming looks buffered through the web tier"

rm -f "$cj"

echo; echo "==================== $pass passed, $fail failed ===================="
exit $(( fail > 0 ))
