{
    "dataset_reader": {
      "type": "srl",
      "bert_model_name": "bert-base-uncased",
    },

    "iterator": {
	"type": "bucket",
	"batch_size": 32,
	"sorting_keys": [["tokens", "num_tokens"]]    
    },

    "train_data_path": std.extVar("SRL_TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("SRL_VALIDATION_DATA_PATH"),

    "model": {
        "type": "srl_bert_bilstm",
        "embedding_dropout": 0.1,
        "bert_model": "bert-base-uncased",
	"recurrent_hidden_layer_size": 768,
	"mlp_hidden_size": 300,
	"predicate_emb_dim": 128
    },

    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            "lr": 5e-5,
	    "t_total": -1,
	    "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 15,
            "num_steps_per_epoch": 8829,
        },
        "num_serialized_models_to_keep": 2,
        "patience": 2,
        "num_epochs": 15,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": [0],
	"should_log_learning_rate": true,
    },

}
