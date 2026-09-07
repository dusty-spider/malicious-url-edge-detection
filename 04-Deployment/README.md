short instruction of the set up
commands on what and how to do:


1.set up virtual environment if needed

2.transfer the files to the edge device by your preffered method via usb or ssh

files needed to run: 
01_pi_predict.py
02_mitmproxy.py
char_to_idx_reproduce.json

adittionally you can download extra deployment option located in the /extra_deployment_option for the pi_api_flask easier set up

3.register your set up with mitmproxy in the browser on the client machine

4.set up the edge device ip as a proxy on the clients browser

5.start the proxy by running: mitmproxy -s mitproxy.py --listen-host 0.0.0.0 --listen-port 8080

