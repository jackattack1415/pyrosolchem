## pyrosolchem
(last updated by Jack Hensley on Feb 19, 2021)

pyrosolchem is a python package to be used for routine chemical analysis
with mass spectrometry and nuclear magnetic resonance spectroscopy. the
primary usage of this package is to process "raw" ms or nmr data and
convert it to "treated" data that can then be plotted or simply exported as
a csv.

~to run the data processing and figure generation for any
research papers that used this package, please install
(step B) and run scripts (step C).~

### A. DESCRIPTION
#### the package is organized according to the following structure:
1. "data_raw" is where the user's raw ms or nmr data should be stored.
2. "conf" contains .yml files, in which experimental runs, their 
descriptions, and corresponding filenames are stored. this is a place
where the user should interface with the package. also, this is where
critical information about the chemicals and constants should be placed.
3. "data_treated" is where the user's treated data is to be located.
the user specifies on the conf .yml file which .csvs are desired for 
further processing or plotting.
4. "notebooks" is where the jupyter notebooks are stored.
5. "src" is where the functions are stored and organized. scripts are 
also located here, which call functions to treat data and generate
plots. this is a place where the user should interface with the 
package.

#### ~CoNtExT && PhiLoSoPhY~
the purpose, as briefly mentioned above, of this package is to
facilitate the conversion of "raw" data to a usable form, usually
concentration (as this is meant for chemical analysis). i rely on
internal standards in measurements that are used to convert signal
to either concentration or relative concentrations. therefore, an
internal standard must be included, alongside the analyte, for all
measurements.

i recognize that the structure is a little convoluted and am interested
in making it more user friendly. the basic methodology is geared toward
preserving the data at each step of the treatment process. the data therefore
is treated in the following order:
- "raw" = input data, untreated
- "filtered" = row-removed data, according to queries
- "processed" = column-removed and calibrated data, according to queries
and functions
- "clustered" = clustered data, using KMeans, according to time of
measurement
- "modeled" = artificial data, synthesized from e.g., lmfit
- "predicted" = artificial data, synthesized from an ode

#### reviewers: a note for you
make_bulk_droplet_paper_data.py, make_chemical_regimes_paper_data.py,
plot_bulk_droplet_paper_data.py, and plot_chemical_regimes_paper_data.py
can simply be run (instructions below) and already are preconfigured 
with the bulk_droplet_experiment.yml and chemical_regimes_experiment.yml files. 
all necessary raw data and treated data are located in their respective places. 
the "make [...] data" files should produce new treated data, and the "plot [...] 
data" files should produce figures in the "results" folder. these are the data 
and figures for two Hensley et al papers, forthcoming. see C for step-by-step
instructions.

### B. INSTALLATION
For installation and setup, assuming that you are on an Ubuntu system. 
To start with you should update and upgrade your apt-get package manager.

```
sudo apt-get update
sudo apt-get upgrade -y
```

Now you can install python 3.6 and the corresponding pip version. 
To do this run the following commands in order:

```
sudo apt-get install python3.6
sudo apt-get install python3-pip
sudo apt-get install python3.6-dev
```

The next step is to set up a virtual environment for managing 
all the packages needed for this project. Run the following 
commands to install virtualenv:

```
python3.6 -m pip install virtualenv
```

With virtualenv installed we can now create a new environment 
for our packages. To do this we can run the following commands 
(where 'my_env' is your chosen name for the environment):

```
python3.6 -m virtualenv my_env
```

you need to activate this environment so that all the necessary 
packages are present. To do this you can run the command below:

```
source my_env/bin/activate
```

Run this command now because in the next step we will clone the 
repo and download all the packages we need into the 'my_env' environment.

```
cd ~
git clone https://github.com/jackattack1415/pyrosolchem
```

All of the required packages for this project are in the 'requirements.txt' file. 
To install the packages run the following commands:

```
cd pyrosolchem/
sudo apt-get install -y libsm6 libxext6 libxrender-dev libpq-dev
pip install -r requirements.txt
pip install psycopg2
```

### C. RUNNING THE PACKAGE
___
1. "Revisiting the reaction of dicarbonyls in aerosol proxy solutions containing 
ammonia: the case of butenedial" Hensley, et al. 2021.

all that is needed to do is the following commands in your terminal:

```
python src/make_chemical_regimes_paper_data.py
```

once that is done, then new csvs should populate the
data_treated folder with format YYYYMMDD_[exp_name]_[data_treatment].csv

then, the following code can be run:

```
python src/plot_chemical_regimes_paper_data.py
```

new figures should populate results/figs_out/

you can use a text editor to directly edit the 
chemical_regimes_experiments.yml file, e.g., to change
the processed data file name for a butenedial/nhx experiment.

2. "Title not yet decided" Hensley, et al. 2021.

all that is needed to do is the following commands in your terminal:

```
python src/make_bulk_droplet_paper_data.py
```

once that is done, then new csvs should populate the
data_treated folder with format YYYYMMDD_[exp_name]_[data_treatment].csv

then, the following code can be run:

```
python src/plot_bulk_droplet_paper_data.py
```

new figures should populate results/figs_out/

you can use a text editor to directly edit the 
bulk_droplet_experiments.yml file, e.g., to change
the processed data file name for a butenedial/nhx experiment.
