from fastapi import FastAPI, UploadFile, File, HTTPException
from extract_one import analyze_pcap_file

app = FastAPI()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        result = await analyze_pcap_file(file)
        return {"suspicious_mac_addresses": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
