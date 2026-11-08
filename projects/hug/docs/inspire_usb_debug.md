# Inspire hand over USB serial

The hand talks RS-485 through a CH340 USB adapter. On a fresh Ubuntu or Jetson image the port usually fails
for one of three reasons: `brltty` grabs the CH340, `ModemManager` probes it, or your user is not in
`dialout`. Fix all three once per machine:

```bash
sudo apt-get purge -y brltty
sudo systemctl mask brltty.path brltty.service 2>/dev/null || true
sudo systemctl stop ModemManager && sudo systemctl disable ModemManager
sudo usermod -aG dialout $USER
sudo udevadm control --reload-rules
# replug the hand, log out and back in (or `newgrp dialout`), then:
ls -l /dev/serial/by-id/
```

`/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` should appear and still be there 30 s later. It is the
default `G1InspireConfig.right_hand_serial_port`. The CH340 reports no serial number, so two adapters share
that name: with two hands, use `/dev/serial/by-path/...` and keep each hand on the same USB socket.

Then check the hand alone, no robot needed:

```bash
uv run --extra real-cu128 python scripts/real/check_hand.py --mode read
```

If it still fails, run these in order right after replugging; the kernel log names the layer that broke.

| Command | What you learn |
|---|---|
| `lsusb` | Is the adapter enumerated? If not: cable, port or power. |
| `ls /dev/ttyUSB*` | Did the `ch341` driver bind? |
| `sudo dmesg -H \| tail -30` | Did something take the port (`brltty`, `ModemManager`)? |
| `ls -l /dev/ttyUSB0` and `groups` | Can your user open it (`dialout`)? |
