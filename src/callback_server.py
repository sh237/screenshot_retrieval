"""
コールバックサーバを定義するモジュール
"""
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import json

def start(conf, embedding, callback):
    """
    コールバックサーバを起動する
    """
    fapi = FastAPI()

    @fapi.post("/oneshot", responses={200: {"content": {"application/json": {"example": {}}}}})
    def execute_oneshot(
        query: UploadFile = File(..., description="検索クエリを含むJSONファイル"),
    ):
        """
        入力されたqueryとデータベース上で保存された文書のsimilarity scoreをTIRで演算して返すAPI

        - input:
            query: 対象物体の操作指示を含む命令文.fieldとして、instructionとinstruction_chatgptを持つ.
        - output:
            JSON形式
            {
                "rank": [rank1_index, rank2_index, ... rankN_index]
            }
        """
        #query.fileは<tempfile.SpooledTemporaryFile object at 0x7f110967ed90>型
        file_contents = query.file
        #file_contentsは<tempfile.SpooledTemporaryFile object at 0x7f110967ed90>型
        file_contents = file_contents.read()
        query = json.loads(file_contents.decode("utf-8"))["query"]
        ret = callback(embedding, query)
        print("score:", ret)
        return JSONResponse(content={"score": ret})

    host_name = conf["hostname"]
    port_num = int(conf["port"])
    
    uvicorn.run(fapi, host=host_name, port=port_num)
