# Setting Up ngc-lava

Setting up ngc-lava is fairly straightforward. The only part that takes some 
time is the setting up of the lava environment itself.

## Installation and Setup Steps

1. To set up and use ngc-lava, first download lava-nc
   found [here](https://lava-nc.org/lava/notebooks/in_depth/tutorial01_installing_lava.html).
2. Install ngc-learn via pip `pip install ngc-learn`
3. Clone ngc-lava and add it as a project source 
   ```bash 
   git clone https://github.com/NACLab/ngc-lava.git  
   pip install -e ngc-lava
   ```
4. To compile for lava, Jax <b>must</b> be turned off; to do this, set the flag 
   `packages/use_base_numpy` to `true` in the ngc-learn 
   `config.json` file. If you do not have a `config.json` file written, the 
   script below will make one for you and add the needed Lava configuration flag:

   ```bash 
   mkdir json_files
   touch json_files/config.json
   echo "{\n  \"packages\": {\n    \"use_base_numpy\": true \n  }\n}" > json_files/config.json
   ```
5. You are now set up and ready to use ngc-lava.