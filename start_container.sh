dig +short myip.opendns.com @resolver1.opendns.com > ip.txt
singularity exec --nv -B /fast_scratch -B /data /fast_scratch/triumfmlutils/containers/base_ml_recommended.simg /bin/bash
