from diagrams import Diagram, Edge
from diagrams.onprem.client import Client
from diagrams.aws.network import CloudFront
from diagrams.generic.compute import Rack
from diagrams.onprem.compute import Server
from diagrams.generic.network import Router

with Diagram("Application Layer", show=False, filename="../out/07_application_protocols_001", graph_attr={"fontsize": "40", "label": "Application layer(7)      HTTP    HTTPS \n Transport layer(4)             80         443"}):
    computer = Client("Computer")
    router = Router("Router")
    isp = Server("ISP")
    cloud1 = CloudFront("Internet")
    cloud2 = CloudFront("Internet")
    server = Rack("Server")

    computer >> Edge(color="orange", fontsize="25") >> router
    computer >> Edge(dir="both") <<  server
    router >> Edge(color="orange", fontsize="25")  >> isp
    isp  >> Edge(color="orange", fontsize="25")  >> cloud1
    cloud1 >> Edge(color="red", fontsize="25") >> cloud2
    cloud2  >> Edge(color="orange", fontsize="25")  >> server

with Diagram("Application Layer", show=False, filename="../out/07_application_protocols_002", graph_attr={"fontsize": "40", "label": "HTTPS 443 \n SSL TLS"}):
    computer = Client("Computer")
    router = Router("Router")
    isp = Server("ISP")
    cloud1 = CloudFront("Internet")
    cloud2 = CloudFront("Internet")
    server = Rack("Server")

    computer >> Edge(color="orange", fontsize="25") >> router
    computer >> Edge(dir="both") <<  server
    router >> Edge(color="orange", fontsize="25")  >> isp
    isp  >> Edge(color="orange", fontsize="25")  >> cloud1
    cloud1 >> Edge(color="red", fontsize="25") >> cloud2
    cloud2  >> Edge(color="orange", fontsize="25")  >> server
