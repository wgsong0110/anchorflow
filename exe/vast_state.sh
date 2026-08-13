#!/usr/bin/env bash
# What state is an instance in, and is it one worth waiting for?
#
# vast.ai puts an instance into "scheduling" when it has accepted the contract
# but has no host to run it on. Nothing about that resolves on a timetable: it
# can sit there indefinitely while the disk is billed, and `start` answers
# "Required resources are currently unavailable, state change queued" rather
# than failing. Both of those are the same condition and both mean destroy and
# rent elsewhere -- waiting has never once paid off.
#
#   vast_state.sh <instance_id>            -> prints the state, exit 0
#   vast_state.sh <instance_id> --waitable -> exit 0 if worth waiting for,
#                                             1 if it should be destroyed,
#                                             2 if the API itself is unreachable
set -uo pipefail
ID=$1
WAITABLE=${2:-}

# an unreachable control plane is not an instance problem, and destroying on it
# would throw away a machine that is very likely still fine
if ! curl -s -o /dev/null --max-time 20 https://console.vast.ai/api/v0/ 2>/dev/null; then
    [ -n "$WAITABLE" ] && exit 2
    echo "api-unreachable"; exit 0
fi

STATE=$(timeout 60 vastai show instances-v1 --raw 2>/dev/null | python3 -c "
import json, sys
try:
    for i in json.load(sys.stdin).get('instances', []):
        if i['id'] == $ID:
            # actual_status is the one that carries 'scheduling'; cur_state and
            # intended_status only ever say running/stopped
            print(i.get('actual_status') or i.get('cur_state') or 'unknown')
            break
    else:
        print('gone')
except Exception:
    print('unknown')
")
[ -z "$WAITABLE" ] && { echo "$STATE"; exit 0; }
case "$STATE" in
    scheduling|gone) exit 1 ;;      # no host, or no longer ours
    *) exit 0 ;;
esac
