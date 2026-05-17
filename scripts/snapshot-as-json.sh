#!/usr/bin/env bash
# Snapshot model cards + datasheets from a running Patra REST API as JSON.
# Run from anywhere. Requires curl, jq, and a Tapis token (for private records).
#
# Usage:
#   PATRA_API_URL=https://patra.pods.icicleai.tapis.io \
#   TAPIS_TOKEN=<paste-token> \
#   ./scripts/snapshot-as-json.sh
#
# Output: ~/patra-backups/snapshot-YYYYMMDD-HHMMSS/
#   modelcards.json                     # list summary
#   datasheets.json                     # list summary
#   modelcards/<uuid>.json              # full detail per record
#   datasheets/<uuid>.json              # full detail per record
set -euo pipefail

: "${PATRA_API_URL:?Set PATRA_API_URL (e.g. https://patra.pods.icicleai.tapis.io)}"
TOKEN="${TAPIS_TOKEN:-}"

OUT="${OUT_DIR:-$HOME/patra-backups/snapshot-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT/modelcards" "$OUT/datasheets"

auth=()
if [[ -n "$TOKEN" ]]; then
  auth=(-H "X-Tapis-Token: $TOKEN")
  echo "Authenticated mode (private records included)"
else
  echo "Anonymous mode (public records only)"
fi

echo "Fetching list endpoints..."
curl -sS --fail "${auth[@]}" "$PATRA_API_URL/modelcards" -o "$OUT/modelcards.json"
curl -sS --fail "${auth[@]}" "$PATRA_API_URL/datasheets"  -o "$OUT/datasheets.json"

for f in "$OUT/modelcards.json" "$OUT/datasheets.json"; do
  if ! jq empty "$f" >/dev/null 2>&1; then
    echo "ERROR: $f is not valid JSON. First 300 bytes:" >&2
    head -c 300 "$f" >&2; echo >&2
    echo "Likely causes: wrong PATRA_API_URL, expired/invalid TAPIS_TOKEN, or pod down." >&2
    exit 1
  fi
done

mc_count=$(jq 'length' "$OUT/modelcards.json")
ds_count=$(jq 'length' "$OUT/datasheets.json")
echo "List: $mc_count model cards, $ds_count datasheets"

echo "Fetching model card details..."
jq -r '.[] | (.uuid // .id // .mc_id // .external_id | tostring)' "$OUT/modelcards.json" \
  | while read -r ident; do
  [[ -z "$ident" || "$ident" == "null" ]] && continue
  curl -sS --fail "${auth[@]}" "$PATRA_API_URL/modelcard/$ident" \
    -o "$OUT/modelcards/$ident.json" || echo "  WARN: failed $ident"
done

echo "Fetching datasheet details..."
jq -r '.[] | (.uuid // .id // .ds_id // .external_id // .identifier | tostring)' "$OUT/datasheets.json" \
  | while read -r ident; do
  [[ -z "$ident" || "$ident" == "null" ]] && continue
  curl -sS --fail "${auth[@]}" "$PATRA_API_URL/datasheet/$ident" \
    -o "$OUT/datasheets/$ident.json" || echo "  WARN: failed $ident"
done

echo
echo "Saved to: $OUT"
echo "  $(find "$OUT/modelcards" -type f | wc -l | tr -d ' ') model card details"
echo "  $(find "$OUT/datasheets" -type f | wc -l | tr -d ' ') datasheet details"
