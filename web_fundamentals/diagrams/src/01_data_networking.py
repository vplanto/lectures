from diagrams import Diagram
from diagrams.generic.device import Tablet
from diagrams.ibm.data import Cloud
from diagrams.ibm.applications import AppServer

with Diagram("Data Networking", show=False, filename="../out/01_data_networking"):
    tablet = Tablet("https://www.wikipedia.org/")
    cloud = Cloud("Cloud")
    app = AppServer("Wikipedia host")
    tablet >> cloud
    cloud >> tablet
    app >> cloud
    cloud << app