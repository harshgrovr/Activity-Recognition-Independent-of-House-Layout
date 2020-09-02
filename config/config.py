#!/usr/bin/env python
config = {
"learning_rate" : 1e-2,
"num_epochs" : 30,
"decay" : 1e-5,
"input_dim" : 36,
"hidden_dim" : 32,
"layer_dim" : 1,
"output_dim" : 12,
"seq_dim" : 1,
"batch_size" :1,
"split_ratio" : 0.5,
"num_workers":1,
"accumulation_steps": 3,
"no_of_subset":30 ,
"subset_overlap_length": 2 ,
"resize_width": 120,
"resize_height": 97,
"image_width": 663,
"image_height": 446,
"ActivityIdList" :  [

            {"name": "Meal_Preparation", "id": 0},
            {"name": "Relax", "id": 1},
            {"name": "Eating", "id": 2},
            {"name": "Work", "id": 3},
            {"name": "Sleeping", "id": 4},
            {"name": "Wash_Dishes", "id": 5},
            {"name": "Bed_to_Toilet", "id": 6},
            {"name": "Enter_Home", "id": 7},
            {"name": "Leave_Home", "id": 8},
            {"name": "Housekeeping", "id": 9},
            {"name": "Respirate", "id": 10},
            {"name": "Idle", "id": 11}
        ]
}