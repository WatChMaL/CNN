########################################################
# 
# Start a jupyter notebook and tell user where to connect
#
# Adapted from Patrick de Perio's script by W.Fedorko
#
########################################################

# Necessary for Compute Canada systems
unset XDG_RUNTIME_DIR

thishost=localhost
jupyter-notebook --no-browser --ip=$thishost --notebook-dir=$PWD >& jupyter_logbook.txt &

sleep 5
echo ""
echo ""

echo "_________________________________________________________________________"
echo "**    FOLLOW THE INSTRUCTIONS BELOW BELOW TO SET UP AN SSH TUNNEL      **"
echo "_________________________________________________________________________"
echo ""

python print_instructions.py 
