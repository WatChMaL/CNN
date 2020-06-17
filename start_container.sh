dig +short myip.opendns.com @resolver1.opendns.com > ip.txt
singularity exec --nv -B /fast_scratch -B /data /fast_scratch/triumfmlutils/containers/baseml_v1.0.6.sif /bin/bash
