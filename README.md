# flowexplore

Interactive visualisation of flow cytometry data + statistics.

Bachelor work, FIIT STU Bratislava

## Running the app locally

### Windows

(make sure, you have [git](https://git-scm.com/download/win) installed)

1. download [python 3.7.3](https://www.python.org/downloads/windows/) and make sure to check "add python 3.7.3 to your 
PATH" (or see adding python to your 
[path](https://datatofish.com/add-python-to-windows-path/))

1. download [Anaconda](https://www.anaconda.com/distribution/), python 3.7 version for Windows

1. navigate to the folder, where you want to download the app, open git console
(right click and select "git bash here") and type 

    ```
    git clone https://github.com/Pomichal/flowexplore_bokeh.git
    ```

1. run Anaconda prompt (hit windows button and search after "anaconda-prompt", navigate to the folder,
 where you cloned the app (into the folder, 
where requirements.txt is placed). 
Find the absolute path to the folder and type into the console:

    ```
    cd /path/to/the/directory
    ```

1. run 
    ```
    conda create --name flowexplore_env --file requirements.txt
    ```

1. run 
    ```
    activate flowexplore_env
    ```

1. navigate to the parent directory
    ```
    cd ..
    ```

1. run
    ```
    bokeh serve flowexplore_bokeh --show
    ```

1. to close the server hit ctrl+C

