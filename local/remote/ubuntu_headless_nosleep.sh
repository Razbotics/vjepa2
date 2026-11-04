#!/usr/bin/env bash
# Keep an Ubuntu desktop awake and reachable over SSH with no monitor attached.
#
#   sudo bash local/remote/ubuntu_headless_nosleep.sh
#
# What it does (all idempotent):
#   1. masks the systemd sleep/suspend/hibernate targets
#   2. tells logind to ignore idle time and power/suspend keys
#   3. stops the GDM login screen from suspending after 20 min (GNOME default on AC)
#   4. disables suspend + screen blanking in the invoking user's GNOME session
# logind settings take effect after a reboot (or `systemctl restart systemd-logind`,
# which can drop the current graphical session, so we do not do it here).
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "run with sudo" >&2
    exit 1
fi
TARGET_USER="${SUDO_USER:-}"

echo "== 1. mask sleep targets"
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

echo "== 2. logind drop-in"
mkdir -p /etc/systemd/logind.conf.d
cat > /etc/systemd/logind.conf.d/90-headless-nosleep.conf <<'EOF'
[Login]
IdleAction=ignore
HandleSuspendKey=ignore
HandleHibernateKey=ignore
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
EOF
cat /etc/systemd/logind.conf.d/90-headless-nosleep.conf

gset() {  # gset <user> <schema> <key> <value>
    local user=$1; shift
    id "$user" &>/dev/null && command -v gsettings &>/dev/null || return 0
    # gsettings needs a session bus. Prefer the user's running one (/run/user/<uid>/bus),
    # fall back to dbus-launch (package dbus-x11) if it is installed.
    local bus="unix:path=/run/user/$(id -u "$user")/bus" launcher=()
    if [ ! -S "/run/user/$(id -u "$user")/bus" ]; then
        command -v dbus-launch &>/dev/null || { echo "   $user: no session bus and no dbus-launch; skipping $*"; return 0; }
        launcher=(dbus-launch --exit-with-session)
    fi
    sudo -u "$user" env DBUS_SESSION_BUS_ADDRESS="$bus" "${launcher[@]}" gsettings set "$@" 2>/dev/null \
        && echo "   $user: gsettings set $*" \
        || echo "   $user: could not set $* (no GNOME for this user?)"
}

echo "== 3. GDM login screen: never suspend"
gset gdm org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type "'nothing'"
gset gdm org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type "'nothing'"

if [ -n "$TARGET_USER" ]; then
    echo "== 4. $TARGET_USER's GNOME session: never suspend, never blank"
    gset "$TARGET_USER" org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type "'nothing'"
    gset "$TARGET_USER" org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type "'nothing'"
    gset "$TARGET_USER" org.gnome.desktop.session idle-delay "uint32 0"
    gset "$TARGET_USER" org.gnome.desktop.screensaver lock-enabled "false"
fi

echo "== result"
systemctl is-enabled sleep.target suspend.target hibernate.target hybrid-sleep.target || true
echo "done. Reboot (or restart systemd-logind) for the logind changes to apply."
