# V(isual)DRIAD

## Description
The project is a molecular dynamics simulation that focuses on simulating the electrostatic interactions between charged bodies. Specifically, it simulates the behavior of negatively charged dust grains and the accumulations of positive charge known as ion wakes.

The simulation employs a comprehensive modeling approach to accurately represent the electrostatic interactions of the point charges. It incorporates a screen Yukawa electrostatic force model, which is a well-known approach in plasma physics. This model considers the screening effect due to the surrounding plasma and provides a realistic representation of the electrostatic forces between the charged particles.

Additionally, the simulation incorporates other well-known plasma theories to capture the complexity of the electrostatic interactions. These theories contribute to the accurate modeling of the interactions between the charged bodies, enhancing the fidelity of the simulation results.

While the electrostatic interaction models are based on established theories, other crucial parameters within the simulation are obtained from experimental data published in scientific papers. Parameters such as the dust charge, ion wake charge, and ion wake separation from the dust grains are derived from experimental measurements. By incorporating experimental data, the simulation ensures that the simulation conditions align with real-world observations, enhancing the reliability and applicability of the results.

Furthermore, the project offers a user-friendly graphical user interface (GUI) that allows users to observe the behavior of the particles in real-time. The GUI provides visual representations of the charged bodies' dynamics and interactions, facilitating a deeper understanding of the simulated phenomena. Moreover, users can interact with the GUI to modify simulation parameters dynamically, enabling them to explore different scenarios and study the effects of parameter variations.

To ensure optimal performance, the simulation employs a hybrid approach. For tasks that do not require computationally intensive operations, a MATLAB wrapper is utilized, harnessing MATLAB's ease of use and convenience. On the other hand, computationally demanding tasks are offloaded to a GPU (Graphics Processing Unit). The GPU is programmed using CUDA, a parallel computing platform and application programming interface (API). This approach maximizes computational efficiency, enabling faster execution of the simulation's resource-intensive tasks.

In summary, this project provides a powerful molecular dynamics simulation tool for studying the electrostatic interactions of charged bodies. By incorporating screen Yukawa electrostatic forces and other established plasma theories, along with experimental data-derived parameters, the simulation offers a high level of accuracy and realism. Coupled with the user-friendly GUI and optimized performance through GPU utilization, the project serves as an invaluable resource for researchers and enthusiasts exploring the dynamics of charged particles in a plasma environment.

## Prerequisites

To run this project, you will need the following:

- MATLAB (version 9.12 or R2022a)
- List of MATLAB Toolboxes:
  - Curve Fitting Toolbox (version 3.7)
  - MATLAB Compiler (version 8.4)
  - Parallel Computing Toolbox (version 7.6)

- NVIDIA CUDA Toolkit (version 11.X)

Make sure you have the specified versions of MATLAB and the CUDA Toolkit installed on your system. The project relies on the mentioned MATLAB Toolboxes and CUDA Toolkit for its execution.

## Getting Started

To run the code and use the project, follow these steps:

1. Clone the repository to your local machine.
`git clone https://github.com/your-username/your-project.git`
2. Open MATLAB and navigate to the project directory.
`cd your-local-address\VDRIAD\src`
3. Set up the required MATLAB environment and toolboxes. Ensure that the necessary toolboxes and CUDA toolkit are properly configured and accessible.
4. Within the MATLAB terminal, run the command `appdesigner`
5. Open the file titled `VIONWAKE.mlapp`. This is the main file of the simulation. It will open the app designer interface for you to modify the appearance of the user interface as well as the code
6. Make sure there's an existing `pcry.ptx` file present in the `src` directory. This ptx file has to be generated the first time that the simulation is run on a new device, or after any modifications have been made to the `pcry.cu` file containing the CUDA code. The ptx file is what allows MATLAB to interpret the CUDA-generated code and feed it to the GPU. To generate the ptx file simply run the following command:
`nvcc -ptx pcry.cu` 
7. To run the simulation, click on the play button labeled as "Run" in the designer tab.

Include any additional instructions or details that are specific to your project, such as data setup, sample usage, or troubleshooting tips.

## License

## Acknowledgements

## Contributing

## Contact
Personal email address: jorgeaugusto.martinezortiz@gmail.com
Academic email address: mart2972@msu.edu


