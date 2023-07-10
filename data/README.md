## フォルダの説明
sample フォルダ: スクショのデータ，ラベル(labels.txt)，OCRの書き起こし，整形した書き起こしを保存  
pickle フォルダ：データセットをpickleにして保存．現状は整形した文字列をデータに入れている

## スクリプトの説明
run.sh: sampleフォルダにあるpngファイルをOCRする(tesseractを使用)  

clean_ocr.py: OCRしたテキストをChatGPTを用いて整形  

create_dataset.py: 整形した文字列データ(X)とアノテーションラベル(y)をデータセットとして作成．各画像のラベルをtrain, val, testで6:2:2で分割

data_loader.py: train, val, testで，整形した文字列とラベルのペアをバッチで取り出す

## アノテーションデータの追加方法
sampleフォルダ内に新しいフォルダを作って(ex. 07)，その中にpng形式のスクショと，labels.txtというtxtファイルを作ってそのなかに，改行区切りでラベルを付けてください
