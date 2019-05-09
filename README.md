# flowexplore

Interactive visualisation of flow cytometry data + statistics.

Bachelor work, FIIT STU Bratislava

## Running the app locally

### Windows

(make sure, you have [git](https://git-scm.com/download/win) installed)

1. download [Anaconda](https://www.anaconda.com/distribution/), python 3.7 version for Windows 
(see [help](https://docs.anaconda.com/anaconda/install/windows/))

1. navigate to the folder, where you want to download the app, open git console
(right click and select "git bash here") and type 

    ```
    git clone https://github.com/Pomichal/flowexplore_bokeh.git
    ```
    hit ENTER.
1. run Anaconda prompt (hit windows button and search after "Anaconda prompt", navigate to the folder,
 where you cloned the app (into the folder, 
where requirements.txt is placed). 
Find the absolute path to the folder and type into the console, for example:

    ```
    cd C:\Users\NTB\Desktop\app\flowexplore_bokeh
    ```
    Replace the path in the example with path to your directory. Hit ENTER.
1. In Anaconda prompt run 
    ```
    conda create --name flowexplore_env --file requirements.txt
    ```
    hit Enter. A question will be shown "Proceed(y/n)?" write "y" and hit ENTER. 
    
    Wait until all requirements will be installed.
1. In Anaconda prompt run 
    ```
    conda activate flowexplore_env
    ```
    hit ENTER. "(flowexplore_env)" will show on the beginning
    of the last row
1. navigate to the parent directory with writing
    ```
    cd ..
    ```
    to the Anaconda prompt and hit ENTER.
1. In Anaconda prompt run
    ```
    bokeh serve flowexplore_bokeh --show
    ```
    The app should open in your browser. Don't close the Anaconda prompt 
    until using the app!
1. to close the server go to the Anaconda prompt and hit ctrl+C

