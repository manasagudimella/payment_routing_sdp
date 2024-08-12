# Optimizing Payment Approvals: Dynamic Programming Approach

## Introduction
This repository contains the codebase and associated resources for the paper **"Optimizing Payment Approvals: Dynamic Programming Approach"**. The paper introduces a smart payment system designed to optimize the selection of payment service providers for each transaction, aiming to increase payment approval rates using dynamic programming.

## Description
The project demonstrates an implementation of a dynamic programming approach to optimize payment approvals by selecting the best payment service provider for each transaction. The model balances the traffic allocation among providers to ensure system reliability and maximize total rewards. The effectiveness of this approach is illustrated using simulated data, showing a potential improvement in overall reward by 8.1%.

## Installation

### Clone the Repository
To get started, clone the repository using the following command:

```bash
git clone https://github.com/yourusername/optimizing-payment-approvals.git
```

Navigate to the Project Directory

After cloning, navigate to the project directory:

```bash
cd payment_routing_sdp
```
Install Dependencies

This project uses Python and several libraries. Install the necessary dependencies with:

```bash
pip install .
```

### Usage

After installing the dependencies, you can run the scripts to simulate the payment optimization process.

Running the Optimization
To run the optimization process, execute the following command:

```bash
python main.py
```
This script will execute the dynamic programming algorithm and output the optimized allocation strategy for the payment service providers.

### Paper
For more details on the methodology and results, please refer to the full paper: [Optimizing Payment Approvals: Dynamic Programming Approach](https://www.ijcttjournal.org/archives/ijctt-v72i7p105)
