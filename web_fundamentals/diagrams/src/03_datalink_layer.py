from diagrams import Diagram, Edge
from diagrams.onprem.client import Client
from diagrams.aws.network import CloudFront
from diagrams.generic.compute import Rack
from diagrams.onprem.compute import Server
from diagrams.generic.network import Router

with Diagram("Networking standards", show=False, filename="../out/03_datalink_layer_001"):
    computer = Client("Computer")
    router = Router("Router")
    isp = Server("ISP")
    cloud1 = CloudFront("Internet")
    cloud2 = CloudFront("Internet")
    server = Rack("Server")

    computer >> Edge(color="orange", label="Ethernet", fontsize="25") >> router
    router >> Edge(color="orange", label="Docsis, FTTH", fontsize="25")  >> isp
    isp  >> Edge(color="orange", label="ITU TG.652", fontsize="25")  >> cloud1
    cloud1 >> Edge(color="red", label="mainly Ethernet, WiMAX, other", fontsize="25") >> cloud2
    cloud2  >> Edge(color="orange", label="Ethernet", fontsize="25")  >> server    

with Diagram("Datalink Layer", show=False, filename="../out/03_datalink_layer_002", graph_attr={"fontsize": "40"}):
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