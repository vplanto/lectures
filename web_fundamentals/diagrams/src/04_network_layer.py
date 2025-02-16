from diagrams import Diagram, Edge
from diagrams.onprem.client import Client
from diagrams.aws.network import CloudFront
from diagrams.generic.compute import Rack
from diagrams.onprem.compute import Server
from diagrams.generic.network import Router

with Diagram("Network protocols", show=False, filename="../out/04_network_layer_001"):
    computer = Client("Computer")
    router = Router("Router")
    isp = Server("ISP")
    cloud1 = CloudFront("Internet")
    cloud2 = CloudFront("Internet")
    server = Rack("Server")

    computer >> Edge(color="orange") >> router
    computer >> Edge(dir="both") << isp
    computer >> Edge(label="IP Addressing & Routing", dir="both") >> server
    router >> Edge(color="orange")  >> isp
    isp  >> Edge(color="orange")  >> cloud1
    isp >> Edge(dir="both") <<  server
    cloud1 >> Edge(color="red") >> cloud2
    cloud2  >> Edge(color="orange")  >> server    

with Diagram("Network Layer", show=False, filename="../out/04_network_layer_002", graph_attr={"fontsize": "40"}):
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