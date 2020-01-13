import tokenization_vn
tokenizer = tokenization_vn.FullTokenizer(
      vocab_file="config/vocab_vn.txt", do_lower_case=False)
tokens = tokenizer.tokenize("Ông trở_thành phụ_tá tiểu_đoàn năm 1909 và sau đó là sĩ_quan kỹ_sư tại Fort_Leavenworth vào năm 1910 .")
print(tokenizer.convert_tokens_to_ids(tokens))