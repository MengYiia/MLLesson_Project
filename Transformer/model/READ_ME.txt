1. best_epoch_model_96 :  decoder_input = 1;
2. best_epoch_model_96_newStart : decoder_input = 拼接encoder_input的最后一个数据;
3. best_epoch_model_96_newnewStart : decoder_input输入拼接encoder_input的最后一个数据, encoder_input仅输入0-94;
4. best_epoch_model_96_newnewnewStart : decoder输入拼接一个全-1的开始符号;
5. best_epoch_model_96_newnewnewStart_conv : decoder输入拼接一个全-1的开始符号 + ffn改为conv;
