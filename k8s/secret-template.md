# Secret template for the MiniMax API key.
#
# This file is intentionally NOT applied directly — the API key should
# be injected at deploy time from GitHub Actions secrets, or pasted
# manually with kubectl. Never commit a real key.
#
# To create the secret manually:
#   kubectl create secret generic job-search-agent-secrets \
#     --namespace marcarndt \
#     --from-literal=minimax-api-key='sk-YOUR_KEY_HERE'
#
# To verify:
#   kubectl get secret job-search-agent-secrets -n marcarndt -o jsonpath='{.data.minimax-api-key}' | base64 -d
