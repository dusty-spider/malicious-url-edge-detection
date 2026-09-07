from mitmproxy import http
from pi_predict import predict_url

#config
BLOCK_MALICIOUS = True          # set False to log only, without blocking
LOG_BENIGN      = False         # set True to also log safe URLs (noisy)


BLOCK_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Blocked</title>
    <style>
        body {{ font-family: Arial, sans-serif; text-align: center; padding: 60px; background: #1a1a1a; color: white; }}
        h1   {{ color: #e74c3c; font-size: 3em; }}
        p    {{ font-size: 1.2em; color: #ccc; }}
        code {{ background: #333; padding: 4px 8px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Blocked</h1>
    <p>This URL was classified as <strong>malicious</strong> by the edge device.</p>
    <p><code>{url}</code></p>
    <p>Confidence: <strong>{confidence:.1%}</strong> &nbsp;|&nbsp; Latency: <strong>{latency_ms}ms</strong></p>
</body>
</html>
"""


class MaliciousURLInterceptor:

    def request(self, flow: http.HTTPFlow) -> None:
        url = flow.request.pretty_url

        result = predict_url(url)

        if result["verdict"] == "malicious":
            print(
                f"[BLOCKED] {url} | "
                f"confidence={result['confidence']:.4f} | "
                f"latency={result['latency_ms']}ms"
            )
            if BLOCK_MALICIOUS:
                flow.response = http.Response.make(
                    403,
                    BLOCK_HTML.format(
                        url=url,
                        confidence=result["confidence"],
                        latency_ms=result["latency_ms"],
                    ),
                    {"Content-Type": "text/html"},
                )

        elif LOG_BENIGN:
            print(
                f"[ALLOWED] {url} | "
                f"confidence={result['confidence']:.4f} | "
                f"latency={result['latency_ms']}ms"
            )


# register the addon with mitmproxy
addons = [MaliciousURLInterceptor()]
