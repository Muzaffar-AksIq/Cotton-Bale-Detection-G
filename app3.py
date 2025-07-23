import pandas as pd
from flask import Flask, request, jsonify
import os
from datetime import datetime, timedelta

CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'detections_log.csv'))

app = Flask(__name__)

# Helper: Load CSV
def load_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    return pd.read_csv(CSV_PATH)

# === 1. LIVE FEED API ===
@app.route('/api/camera/live', methods=['GET'])
def camera_live():
    df = load_data()
    if df.empty:
        return jsonify({"cameras": []})

    cameras = []
    for camera_id, group in df.groupby("CameraID"):
        last_row = group.iloc[-1]
        # Consider "active" if last update within 30s, else "offline"
        last_time = pd.to_datetime(last_row["Timestamp"])
        now = datetime.utcnow()
        status = "active" if (now - last_time) < timedelta(seconds=30) else "offline"
        cameras.append({
            "cameraId": camera_id,
            "cameraName": last_row["CameraName"],
            "lineCount": int(last_row["LineCount"]),
            "lastUpdated": last_row["Timestamp"],
            "status": status
        })
    return jsonify({"cameras": cameras})

@app.route('/api/camera/history', methods=['GET'])
def camera_history():
    camera_id = request.args.get("cameraId")
    limit = int(request.args.get("limit", 100))

    df = load_data()
    if df.empty or camera_id is None:
        return jsonify({"cameraId": camera_id or "", "history": []})

    df["CameraID"] = df["CameraID"].astype(str)   # <--- FIX

    df = df[df["CameraID"] == camera_id].copy()
    df = df.sort_values(by="Timestamp", ascending=False).head(limit)

    history = []
    for idx, row in df.iterrows():
        history.append({
            "entryId": f"entry_{idx}",
            "timestamp": row["Timestamp"],
            "lineNumber": int(row["LineCount"]),
            "anomalyDetected": str(row.get("AnomalyDetected", "false")).lower() == "true",
            "anomalyType": row.get("AnomalyType") if pd.notnull(row.get("AnomalyType")) and str(row.get("AnomalyType")).strip() else None
        })

    history = sorted(history, key=lambda x: x["timestamp"], reverse=True)

    return jsonify({"cameraId": camera_id, "history": history})


# === 3. ANOMALY SUMMARY API ===
@app.route('/api/camera/anomalies', methods=['GET'])
def anomaly_summary():
    from_date = request.args.get("fromDate")
    to_date = request.args.get("toDate")

    df = load_data()
    if df.empty:
        return jsonify({"fromDate": from_date, "toDate": to_date, "cameras": []})

    # Filter by date
    df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
    from_dt = datetime.strptime(from_date, "%Y-%m-%d").date() if from_date else None
    to_dt = datetime.strptime(to_date, "%Y-%m-%d").date() if to_date else None
    if from_dt: df = df[df["Date"] >= from_dt]
    if to_dt: df = df[df["Date"] <= to_dt]

    cameras = []
    for camera_id, group in df.groupby("CameraID"):
        total_entries = group.shape[0]
        anomaly_rows = group[group["AnomalyDetected"].astype(str).str.lower() == "true"]
        anomalies_detected = anomaly_rows.shape[0]
        anomaly_breakdown = dict(anomaly_rows["AnomalyType"].value_counts())

        cameras.append({
            "cameraId": camera_id,
            "totalEntries": total_entries,
            "anomaliesDetected": anomalies_detected,
            "anomalyBreakdown": anomaly_breakdown
        })

    return jsonify({
        "fromDate": from_date,
        "toDate": to_date,
        "cameras": cameras
    })

if __name__ == "__main__":
    app.run(port=7864)
