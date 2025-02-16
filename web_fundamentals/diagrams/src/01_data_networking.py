from diagrams import Diagram
from diagrams.onprem.client import Client
from diagrams.aws.network import CloudFront
from diagrams.generic.compute import Rack

with Diagram("Data Networking", show=False, filename="../out/01_data_networking_001"):
    pc = Client("Personal Computer")
    cloud = CloudFront("Cloud Network")
    server = Rack("Server")

    pc >> cloud >> server

from diagrams.onprem.client import User
from diagrams.generic.device import Mobile
from diagrams.ibm.network import TransitGateway

with Diagram("Mobile communication", show=False, filename="../out/01_data_networking_002"):
    client1 = User("Alice")
    smartphone1 = Mobile("Smartphone 1")
    tower1 = TransitGateway("Tower 1")
    tower2 = TransitGateway("Tower 2")
    client2 = User("Bob")
    smartphone2 = Mobile("Smartphone 2")

    client1 >> smartphone1 >> tower1 >> tower2 >> smartphone2 >> client2