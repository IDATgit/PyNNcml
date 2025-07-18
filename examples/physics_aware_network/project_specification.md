# Rain Estimation from CML Data

We want to estimate rain rate from Commercial Microwave Links (CML) using their Transmitted Signal Level (TSL) and Received Signal Level (RSL) signals. We will take a data-driven approach and train a neural network to perform this translation.

## Data

We are working with OpenMRG, a dataset collected in Sweden.

**OpenMRG Quick Facts:**
- **Observation Period:** June–August 2015 (summer season, including heavy convective events).
- **Region:** Greater Gothenburg, south-west Sweden (≈ 57.6°N, 11.9°E).
- **Sensors Provided:**
    - 271 commercial microwave sub-links (RSL & TSL every 10s, 6–38 GHz, 0.1–7 km path lengths).
    - 46 tipping-bucket rain gauges (0.1 mm tips, 1 min logging).
    - 1 SMHI reference gauge (1 min).
    - C-band weather radar composite (240×240 pixels, 1 km resolution, 5 min scans).
- **Public Release:** Zenodo (DOI: 10.5281/zenodo.7107689), CC-BY-4.0 license.
- **Pre-processing in PyNNcml:**
    - Radar reflectivity is converted to rain rate (Z = 200R¹·⁵) and averaged to 15 min.
    - Gauges are converted to a 15 min rain-rate using a 15-minute moving window.
    - RSL/TSL data remains at its native 10s resolution but is folded into 15-minute blocks (90 samples) to align with the targets.

These harmonised series (rain gauge, radar, and CML attenuation) are what the neural network pipeline in this project will ingest for training and validation.

## Model Architecture

![Model Architecture](documents/model_architecture.png)

The idea is to separate the task into two relatively simple models. The first model will evaluate the baseline attenuation signal (which includes both the constant free-space attenuation and any non-rain related attenuation). The output of this model will then be subtracted from the original attenuation signal.

The next step is to convert the compensated attenuation signal to a rain rate. Given the metadata (link frequency, length, etc.), this is a scalar conversion where each time sample is processed individually. This approach will hopefully yield easily interpretable models.

### Attenuation Baseline Removal Architecture

**Inputs:**
- Attenuation signal (1D array)
- Link metadata (frequency, polarization, etc.)

**Outputs:**
- Attenuation signal (1D array)

We will use a simple CNN-based architecture with a StyleGAN-like approach for metadata fusion.

```mermaid
graph TD;
    %% Define node groups (subgraphs) to organize the view
    subgraph " "
        direction LR
        subgraph "Metadata & Style Path"
            MetaIn["Metadata Input<br/>[B x 32]"] --> MLP["Metadata MLP"] --> Styles["Style Vectors"];
        end
    end

    subgraph "Signal Processing Path"
        A["Input Signal<br/>[1 x 958]"]
        --> Conv1["Conv1D (k=3)<br/>[32 x 958]"]
        --> AdaIN1["AdaIN (C=32)"]
        --> Conv2["Conv1D (k=5)<br/>[64 x 958]"]
        --> AdaIN2["AdaIN (C=64)"]
        --> Conv3["Conv1D (k=15)<br/>[64 x 958]"]
        --> AdaIN3["AdaIN (C=64)"]
        --> Conv4["Conv1D (k=100)<br/>[128 x 958]"]
        --> AdaIN4["AdaIN (C=128)"]
        --> Conv5["Conv1D (k=900)<br/>[128 x 958]"]
        --> AdaIN5["AdaIN (C=128)"]
        --> Conv6["Conv1D (k=1)<br/>[64 x 958]"]
        --> AdaIN6["AdaIN (C=64)"]
        --> Conv7["Conv1D (k=1)<br/>[1 x 958]"]
        --> ReLU1["ReLU"]
        --> Z["Output<br/>[1 x 958]"];
    end

    %% Use invisible links for better layout of style injection
    Styles -.-> AdaIN1;
    Styles -.-> AdaIN2;
    Styles -.-> AdaIN3;
    Styles -.-> AdaIN4;
    Styles -.-> AdaIN5;
    Styles -.-> AdaIN6;
    
    %% Styling
    classDef metadata fill:#A040A0,color:#fff,stroke:#000,stroke-width:2px;
    classDef signal fill:#3670A6,color:#fff,stroke:#000,stroke-width:2px;
    classDef output fill:#2E8B57,color:#fff,stroke:#000,stroke-width:2px;
    classDef relu fill:#FFD700,color:#000,stroke:#000,stroke-width:1px;
    
    class MetaIn,MLP,Styles metadata;
    class A signal;
    class Z output;
    class ReLU1 relu;
```

### Attenuation to Rain Rate Architecture

Here we use a Fully Connected (FC) network with StyleGAN-like metadata fusion.

**Inputs:**
- Attenuation signal (single value)
- Link metadata (frequency, polarization, etc.)

**Outputs:**
- Rain rate (single value)

```mermaid
graph TD;
    %% Metadata & Style Path
    subgraph " "
        direction LR
        subgraph "Metadata & Style Path"
            MetaIn2["Metadata Input<br/>[B x 32]"] --> MLP2["Metadata MLP"] --> Styles2["Style Vectors"];
        end
    end

    subgraph "Attenuation to Rain Rate Path"
        X2["Input Attenuation<br/>[1]"]
        --> FC1["FC Layer 1<br/>[1 x 128]"]
        --> AdaIN_FC1["AdaIN Fusion 1<br/>[128]"]
        --> FC2["FC Layer 2<br/>[128 x 128]"]
        --> AdaIN_FC2["AdaIN Fusion 2<br/>[128]"]
        --> FC3["FC Layer 3<br/>[128 x 128]"]
        --> AdaIN_FC3["AdaIN Fusion 3<br/>[128]"]
        --> FC4["FC Layer 4<br/>[128 x 128]"]
        --> AdaIN_FC4["AdaIN Fusion 4<br/>[128]"]
        --> FC5["FC Layer 5<br/>[128 x 1]"]
        --> AdaIN_FC5["AdaIN Fusion 5<br/>[1]"]
        --> ReLU2["ReLU"]
        --> Y2["Output Rain Rate<br/>[1]"];
    end

    %% Style vector fusion to all AdaINs
    Styles2 -.-> AdaIN_FC1;
    Styles2 -.-> AdaIN_FC2;
    Styles2 -.-> AdaIN_FC3;
    Styles2 -.-> AdaIN_FC4;
    Styles2 -.-> AdaIN_FC5;

    %% Styling
    classDef metadata fill:#A040A0,color:#fff,stroke:#000,stroke-width:2px;
    classDef signal fill:#3670A6,color:#fff,stroke:#000,stroke-width:2px;
    classDef output fill:#2E8B57,color:#fff,stroke:#000,stroke-width:2px;
    classDef relu fill:#FFD700,color:#000,stroke:#000,stroke-width:1px;

    class MetaIn2,MLP2,Styles2 metadata;
    class X2 signal;
    class Y2 output;
    class ReLU2 relu;
```


## Development Methodology
### Step 1 - data inspection
using the pyncnn dataloader, explore the given data.
take a single example:
plot rsl and tsl signal data with the corresponding datetimes. 
plot rain rate data with the corresponding datetimes.
script is implemented in ./data_inspection/data_inspection.py.
### Step 2 - implement the models
as described in the above section implement the two models
1. ./models/attanuation_baseline_removal.py
2. ./models/attanuation_to_rainrate.py
3. ./models/physics_informed_rain_estimation.py (this is a wrapper which include the two models with appropriate connections).
### Step 3 - implement a trainer
create a trainer to train the physics_informed_rain_estimation model. 
the loss function is l2 on rain rate, with proper weightening to account for the imbalance in the dry / wet (there is 10x more dry then wet periods). another regulizer will be added after the attnuation substruction. so that we expect '0' attanuation before the convertion if there is a dry period. 
crucial features for the trainers are:
1. save checkpoints into ./results/checkpoints.
2. print loss function and the two components (before and after scaling). both for test set and train in every epoch
3. generate tensorflow training graphs.
4. keep active window which shows single training example (for example the first one in the first batch). for this sample we want to see a graph of the real rain vs predicted rain. another graph for the original attanuation with the subtracted version.

### step 4 - train
train the model, we will need to stop here and iterate if we don't get a proper result.
we have the loss function and the graph vizualization to get an insight of whats going right or wrong.

### step 5 - analyze results
we want to see the following:
1. graph of 1 random example from the training set with real rain and predicted rain. as well the original and compensated attanuation signal.
2. another graph with 1 random sample from test set.
3. graph for our worst performing training sample
4. graph for our worst performaing test sample

















 

