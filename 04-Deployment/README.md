# Raspberry Pi deployment

Files needed on the Pi:

- `pi_predict.py`
- `02_mitmproxy.py`
- `char_to_idx_reproduce.json`
- one deployed model from `Models/deployed_models/`

Copy the model you want to use onto the Pi and set `MODEL_PATH` in
`pi_predict.py` to match its filename.

Setup:

1. Create a virtual environment if needed.
2. Transfer the files with USB or scp.
3. Register mitmproxy's CA certificate in the browser on the client machine.
4. Set the Pi's IP address as the browser's proxy.

Start the proxy:

    mitmproxy -s 02_mitmproxy.py --listen-host 0.0.0.0 --listen-port 8080

This is a lab setup. The proxy has no authentication and listens on all interfaces,
so do not expose it outside a trusted network.

An optional Flask API prototype is in `extra_deployment_option/`. It was not used in
the final proxy demo.
