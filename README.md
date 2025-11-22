# NYC Vehicle Collisions Interactive Dashboard

An interactive web-based dashboard for exploring and analyzing NYC motor vehicle collision data, built with Python Dash and Plotly.

## 🎯 Features

### Interactive Filters
- **Multiple Dropdown Filters**: Borough, Year, Vehicle Type, Contributing Factor, Injury Type, Person Type
- **Smart Search Mode**: Natural language search (e.g., "Brooklyn 2022 pedestrian crashes")
- **Generate Report Button**: Central button to update all visualizations dynamically



### Real-time Interactivity
- Hover tooltips on all charts
- Zoom and pan capabilities on maps
- Dynamic filtering across all visualizations
- Responsive layout adapting to screen size

# 🚀 Deployment Guide: PythonAnywhere

This guide outlines the steps to deploy the NYC Collisions Dashboard on [PythonAnywhere](https://www.pythonanywhere.com/) (Free Tier).

## 1. Prerequisites
*   A free "Beginner" account on PythonAnywhere.
*   This GitHub repository.
*   *Note:* The free tier has a 512MB disk limit. This guide includes specific steps to stay within that limit.

## 2. Initial Setup (Bash Console)
Open a new *Bash* console on PythonAnywhere and run the following commands:

### A. Clone the Repository
bash
git clone https://github.com/AmrKhaled05/NYC-Collisions-Interactive-Visualization.git
cd NYC-Collisions-Interactive-Visualization


### B. Free Up Disk Space (Critical)
To avoid the Disk quota exceeded error during installation, remove cache and git history:
bash
# Delete pip cache
rm -rf ~/.cache/pip

# Delete git history (saves ~100MB+, but disables git pull)
rm -rf .git

# Delete local share cache
rm -rf ~/.local/share


### C. Create Virtual Environment
bash
# Create environment using Python 3.10
mkvirtualenv --python=/usr/bin/python3.10 myenv

*If mkvirtualenv is not found, run source virtualenvwrapper.sh first.*

### D. Install Dependencies
Install packages without caching to save space:
bash
pip install --no-cache-dir dash pandas plotly dash-bootstrap-components pyarrow


## 3. Web App Configuration

1.  Go to the *Web* tab on the PythonAnywhere dashboard.
2.  Click *Add a new web app*.
3.  Select *Flask* -> *Python 3.10* -> (Next) -> (Next).

### Update Paths
In the "Code" section of the Web tab, update these paths:

*   *Source code:* /home/yourusername/NYC-Collisions-Interactive-Visualization
*   *Working directory:* /home/yourusername/NYC-Collisions-Interactive-Visualization
*   *Virtualenv:* /home/yourusername/.virtualenvs/myenv

*(Replace yourusername with your actual PythonAnywhere username. To get the exact virtualenv path, run cdvirtualenv then pwd in the console).*

## 4. Configure WSGI File
Click the *WSGI configuration file* link (e.g., /var/www/yourname_pythonanywhere_com_wsgi.py) and replace the *entire content* with:

python
import sys
import os

# 1. Add project directory to python path
project_home = '/home/yourusername/NYC-Collisions-Interactive-Visualization'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# 2. Import the Dash app
from app import app

# 3. Expose the server object for PythonAnywhere
application = app.server

*Remember to replace yourusername with your actual username.*

## 5. Data File Setup
Since the dataset (Final_Data.parquet) is large, it might not clone correctly or might be too big for the free tier.

1.  If using a sample file, rename it to Sample.parquet (or ensure your code looks for the correct filename).
2.  Go to the *Files* tab -> NYC-Collisions-Interactive-Visualization.
3.  Use the *Upload a file* button to upload your .parquet file manually if it appears as 0 bytes or 133 bytes.

## 6. Launch
1.  Go back to the *Web* tab.
2.  Click the green *Reload* button.
3.  Visit your URL: https://yourusername.pythonanywhere.com.

## 📊 How to Use

### Basic Usage
1. **Select Filters**: Choose your desired filters from the dropdown menus (Borough, Year, Vehicle Type, etc.)
2. **Click "Generate Report"**: Press the green "Generate Report" button to update all visualizations
3. **Explore**: Interact with charts by hovering, zooming, and panning

### Search Mode
Use natural language queries in the search box:
- "Brooklyn 2022 pedestrian crashes"
- "Manhattan killed cyclists 2023"
- "Queens injured sedan 2021"

The system will automatically detect and apply relevant filters.




## 👥 Team Contributions

### Team Member 1: [Ali Tharwat]
- Persons Data Visualization
- Dataset compression
- User interface design, search bar and slider design.

### Team Member 2: [Mostafa Ahmed]
- Crash data cleaning and preprocessing
- Merged Data cleaning
- Website deployment

### Team Member 3: [Amr Khaled]
- Crashes Data Visualization
- Website development 
- Website visualizations

### Team Member 4: [Mohamed Ghoraba]
- Person data cleaning
- Website development
- Testing and deployment

### Team Member 5: [Ahmed Ramy]
- Data integration and merging
- Research questions formulation
- Map Visualization 

## 🔬 Research Questions

1. Do driver demographics (age, gender) have a correlation with higher crash involvement?

2.Are certain vehicle types (taxis, trucks,etc.) involved in more crashes than others?

3.How did crash rates change during a major event such as COVID-19 lockdowns in 2020 ?

4.How have crash rates evolved from 2012 to 2025, and are there signs of improvement? 

5.Which borough has the highest crash rate per capita?

6.Who are the most type of people involved in crashes?

7.What's the safety equipment involved in most crashes and the safety equipment involved in least crashes?

8.Which specific cross streets consistently report the highest number of crashes?

9.Which contributing factors (e.g., distracted driving, speeding, alcohol) are most strongly associated with fatal crashes versus injury-only crashes?

10.What are the top 5 contributing vehicle factors across all collisions?

## 📝 Data Sources

- **NYC Motor Vehicle Collisions - Crashes**: [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/data_preview)
- **NYC Motor Vehicle Collisions - Person**: [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu/data_preview)


## 📄 License

This project is developed for educational purposes as part of the Data Engineering and Visualization course at the German International University.

## 🙏 Acknowledgments

- German International University - Faculty of Informatics and Computer Science
- Dr. Nada Sharaf
- Teaching Assistants: Mariam Ali, May Magdy, Mohamed Abdelsatar
- NYC Open Data Platform

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the team members through the repository.

---

