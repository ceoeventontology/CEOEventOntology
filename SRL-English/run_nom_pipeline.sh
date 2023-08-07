ID_OUTPUT="$1-id-output.txt"
ID_SRL_INPUT="$1-srl-input.txt"
#SRL_INPUT="$1-srl-input.txt"
SRL_INPUT="$1"

allennlp predict /nas/luka-group/nanx/EventInduction/harddisk/models/SRL-English/nom-id-bert/model.tar.gz $1 --output-file ${ID_OUTPUT} --cuda-device 0 --predictor "nombank-id" --include-package id_nominal
python convert_id_to_srl_input.py ${ID_OUTPUT} ${ID_SRL_INPUT}
if [ -z "$2" ]
then
  echo "no output file specified..."
#	allennlp predict /nas/luka-group/nanx/EventInduction/harddisk/models/SRL-English/nom-srl-bert/model.tar.gz ${SRL_INPUT} --cuda-device 0 --predictor "nombank-semantic-role-labeling" --include-package nominal_srl
	allennlp predict /nas/luka-group/nanx/EventInduction/harddisk/models/SRL-English/nom-sense-srl/model.tar.gz ${ID_SRL_INPUT} --cuda-device 0 --predictor "nombank-sense-srl" --include-package nominal_sense_srl
else
  echo "output file {$2} specified..."
#	allennlp predict /nas/luka-group/nanx/EventInduction/harddisk/models/SRL-English/nom-srl-bert/model.tar.gz ${SRL_INPUT} --output-file $2 --cuda-device 0 --predictor "nombank-semantic-role-labeling" --include-package nominal_srl
	allennlp predict /nas/luka-group/nanx/EventInduction/harddisk/models/SRL-English/nom-sense-srl/model.tar.gz ${ID_SRL_INPUT} --output-file $2 --cuda-device 0 --predictor "nombank-sense-srl" --include-package nominal_sense_srl
fi

