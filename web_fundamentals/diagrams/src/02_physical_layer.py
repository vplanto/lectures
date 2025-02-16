from diagrams import Diagram
from diagrams.onprem.client import Client
from diagrams.aws.network import CloudFront
from diagrams.generic.compute import Rack

from diagrams.generic.network import Router

with Diagram("Home network", show=False, filename="../out/02_physical_layer_001"):
    computer = Client("Computer")
    router = Router("Router")
    computer >> router

with Diagram("Home network and Internet", show=False, filename="../out/02_physical_layer_002"):
    computer = Client("Computer")
    router = Router("Router")
    cloud = CloudFront("Internet")

    computer >> router >> cloud

from diagrams import Edge
from diagrams.onprem.compute import Server

with Diagram("Home network and Internet and Remote Server", show=False, filename="../out/02_physical_layer_003"):
    computer = Client("Computer")
    router = Router("Router")
    isp = Server("ISP")
    cloud1 = CloudFront("Internet")
    cloud2 = CloudFront("Internet")
    server = Rack("Server")

    computer >> Edge(color="orange", label="Cable") >> router
    router >> Edge(color="orange", label="Cable")  >> isp
    isp  >> Edge(color="orange", label="Cable")  >> cloud1
    cloud1 >> Edge(color="red", label="Cable & Wireless") >> cloud2
    cloud2  >> Edge(color="orange", label="Cable")  >> server

with Diagram("Home network and Internet and Remote Server", show=False, filename="../out/02_physical_layer_004"):
    computer = Client("Computer")
    router = Router("Router")
    isp = Server("ISP")
    cloud1 = CloudFront("Internet")
    cloud2 = CloudFront("Internet")
    server = Rack("Server")

    computer >> Edge(color="orange", label="Twisted pair", fontsize="25") >> router
    router >> Edge(color="orange", label="CoAx", fontsize="25")  >> isp
    isp  >> Edge(color="orange", label="Fiber Optics", fontsize="25")  >> cloud1
    cloud1 >> Edge(color="red", label="Fiber Optics & Twisted pair", fontsize="25") >> cloud2
    cloud2  >> Edge(color="orange", label="Twisted pair", fontsize="25")  >> server    

with Diagram("Physical Layer", show=False, filename="../out/02_physical_layer_005", graph_attr={"fontsize": "40"}):
    computer = Client("Computer")
    router = Router("Router")
    isp = Server("ISP")
    cloud1 = CloudFront("Internet")
    cloud2 = CloudFront("Internet")
    server = Rack("Server")

    computer >> Edge(color="orange", fontsize="25") >> router
    router >> Edge(color="orange", fontsize="25")  >> isp
    isp  >> Edge(color="orange", fontsize="25")  >> cloud1
    cloud1 >> Edge(color="red", fontsize="25") >> cloud2
    cloud2  >> Edge(color="orange", fontsize="25")  >> server