# <img src=https://user-images.githubusercontent.com/122404051/235878740-0f447969-b786-41de-93ca-a4528a4db470.gif width="48" height="48" >  DineNavigator : Intelligent Price and Location Recommender
Built a web-based recommendation model using machine learning algorithms to recommend optimal restaurant locations and prices based on user preferences for cuisine, location, and price.
Used Jupyter Notebook to train the machine learning algorithms, and implemented the model using Flask web framework.
Developed a user-friendly web page using HTML and CSS to display the recommendation results.
Utilized libraries such as Scikit-Learn, importnb, and Pandas to perform data processing, modeling, and web development tasks.

<br>

##  <img src="https://user-images.githubusercontent.com/106439762/181935629-b3c47bd3-77fb-4431-a11c-ff8ba0942b63.gif" width="48" height="48"> *Folder Structure Guide*

| Files/Folder| Description |
| ------------- | ------------- |
| *.ipynb Folder* | This folder includes the Jupyter Notebook files that were utilized to make Machine Learning models  |
| *Dataset Folder* | Within this folder, there are two CSV tables that were acquired by scraping data from the web. And used in this project to create ML prediction models  |
| *Presentation Folder* | This folder contains the presentation in pdf format.  |
| *Python Folder* | This folder contains Python file |
| *Web-page Folder* | The contents of this folder comprise HTML and CSS files, which are used to generate our webpage. |
| *Images Folder* | It contains all the files that are used for styling our webpage, like background images |
| *Pickle Folder* | It contains all the Pickle files that Stores the models |

<br>
<p align="center"><img src="https://user-images.githubusercontent.com/122404051/235923506-3e8b5280-f760-44d3-af9b-9da55946b26a.gif"
 width="400" ></p>
 
 ##  <img src=https://user-images.githubusercontent.com/106439762/178803205-47a08ce7-2187-4f96-b301-a2b68690619a.gif width="48" height="48" > Prior Knowledge <br>
<br>
<p align="center"><a><img src="https://user-images.githubusercontent.com/122404051/235928491-09398424-2c7c-45f8-a0d5-f452320d015c.jpg" width="1050" height="35"></a></p>


<br>

## <img src=https://user-images.githubusercontent.com/106439762/178804195-d9db61fb-b2cf-4c8f-bfc3-214cfe0f534c.gif width="48" height="48" > Quick Summary

    1. The Price prediction model is created on Decision Tree and Location prediction model is created on Random Forest as accuracy for both of these was best than others.
  ![DCMODEL](https://github.com/harshp1801/AutoInsights-Used-Car-Analysis-Dashboard/assets/128036066/fcc1a30b-5bff-4f8d-9cff-596d82ff9cc6)
	
  ![RFCMODEL](https://github.com/harshp1801/AutoInsights-Used-Car-Analysis-Dashboard/assets/128036066/f9dbe100-b85f-40df-a5a1-1ecfb8b93661)

    2. Both of these models take three parameters, two of which are provided by the user they are Cuisine and Price For One and it takes the third parameter from the insights that is the Average Price For One of that location
  
  ![InsightsGenerator](https://github.com/harshp1801/AutoInsights-Used-Car-Analysis-Dashboard/assets/128036066/8a7d01ec-6bc1-432f-8ac6-be2234914ac5)
  
  ![avgpriceofarea](https://github.com/harshp1801/Recommender/assets/128036066/3d2a8528-156e-430c-9424-5cf41f623f6b.png)
  
    
    3. All of this inputs are taken from the html landing page using flask and then is compare with pkl file and is later predicted.
  
  	a) HTML code for Main Page
  
  ![Index](https://github.com/harshp1801/Used-Car-Analysis-Dashboard/assets/128036066/2bd02ac6-d244-4ea2-aa3b-a9852c35dc21)

  	b) HTML code for Prediction Page
  
  ![prediction](https://github.com/harshp1801/Used-Car-Analysis-Dashboard/assets/128036066/49670034-ce6c-412d-9d07-9e911bd94e34)
  	
	c) Css Code For both Pages
  
  ![csssfile](https://github.com/harshp1801/Recommender/assets/128036066/63cc22df-03f3-4bbb-8591-bc6bbd3c8eec) 
  
    4. We connected all the Notebooks to a main Python file using a library  'importnb'
    
    5. Later when all the Notebooks were connected, we used flask to deploy the model on Web Page
    
   ![dep1](https://github.com/harshp1801/Recommender/assets/128036066/458dd813-8c00-4381-9206-652983b01aac)
   ![dep2](https://github.com/harshp1801/Recommender/assets/128036066/84af5783-666e-4182-9267-eda8696dc619)
    
    6. Flask was used to Get and post the data on webpage
	
## <img src="https://user-images.githubusercontent.com/122404051/235936187-301b427a-9f69-4c72-8d3e-e289a50c3a59.png" width="48" height="48"/> Landing Page Screenshot
![landing page](https://user-images.githubusercontent.com/122404051/235935721-faca695c-97ea-4591-a633-ee1bfd2a052b.jpg)

## <img src="https://user-images.githubusercontent.com/122404051/235936187-301b427a-9f69-4c72-8d3e-e289a50c3a59.png" width="48" height="48"/> Prediction Page Screenshot
![predction page](https://user-images.githubusercontent.com/122404051/235935916-78179f03-339e-49db-814e-a185e5cd3a2d.jpeg)
