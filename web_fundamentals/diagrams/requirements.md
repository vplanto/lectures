# Project Requirements

## Preprequisites
- ***Python***: Make sure you have Python installed on your system. You can download it from [python.org](https://python.org/downloads/).
- ***Graphviz***: Required for rendering diagrams.

## Installation Steps

### 1. Install GraphvizC

#### On Windows:
- Download the Graphviz installer from the [Graphviz download page](https://graphviz.org/download/).
- Run the installer and follow the instructions
- Add the Graphviz `bin` directory (e.g., `C:\Program Files\Graphviz\bin`) to your system's PATH environment variable.

#### On macOS/Linux:
- Open a terminal and run:
``apt get install graphviz 
``
### 2. Set up VSW - Create a Virtual Environment
1. Open a terminal and navigate to your project directory:
2. Create a virtual environment using the venv module:
``python -m venv venv
``
### 3. Activate the virtual environment:
#### On Windows:
- ``venv\Scripts\activate
``
#### On macOS/Linux:
- ``source venv/bin/activate
``
### 4. Install the diagrams library using pip:
- ``pip install diagrams
``

### 5. Sample Script
- Create a file named sample.py with the following content:
```python
from diagrams import Diagram
from diagrams.k8s.clusterconfig import HPA
from diagrams.k8s.compute import Deployment, Pod, ReplicaSet
from diagrams.k8s.network import Ingress, Service

with Diagram("Exposed Pod with 3 Replicas", show=False, filename="../out/sample.png"):
    net = Ingress("domain.com") >> Service("svc")
    net >> [Pod("pod1"),
            Pod("pod2"),
            Pod("pod3")] << ReplicaSet("rs") << Deployment("dp") << HPA("hpa")
```

- Generating the Diagram
``python sample.py
``