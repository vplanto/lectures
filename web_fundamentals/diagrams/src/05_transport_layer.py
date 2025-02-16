from diagrams import Diagram, Edge
from diagrams.onprem.client import Client
from diagrams.aws.network import CloudFront
from diagrams.generic.compute import Rack
from diagrams.onprem.compute import Server
from diagrams.generic.network import Router

with Diagram("Network protocols", show=False, filename="../out/05_transport_layer_001"):
    computer = Client("Computer")
    router = Router("Router")
    isp = Server("ISP")
    cloud1 = CloudFront("Internet")
    cloud2 = CloudFront("Internet")
    server = Rack("Server")

    computer >> Edge(color="orange") >> router
    computer >> Edge(label="Transmission Control Protocol (TCP)", dir="both") << server
    router >> Edge(color="orange")  >> isp
    isp  >> Edge(color="orange")  >> cloud1
    cloud1 >> Edge(color="red") >> cloud2
    cloud2  >> Edge(color="orange")  >> server    

with Diagram("Transport Layer", show=False, filename="../out/05_transport_layer_002", graph_attr={"fontsize": "40"}):
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