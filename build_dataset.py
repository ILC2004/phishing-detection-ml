import pandas as pd
import zipfile
import os
import email
import bz2

rows = []

# ---------- PHISHING DATA ----------
with zipfile.ZipFile("phishing_raw.zip") as z:
    for file in z.namelist():
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(z.open(file), encoding="latin1")
            except:
                continue

            cols = [c.lower() for c in df.columns]

            if "subject" in cols and "body" in cols:
                df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
            elif "text" in cols:
                df["text"] = df["text"]
            else:
                continue

            if "label" in cols:
                label_col = "label"
            else:
                continue

            for t,l in zip(df["text"], df[label_col]):
                rows.append((str(t), int(l)))

# ---------- HAM DATA ----------
with bz2.open("ham_raw.bz2") as f:
    content = f.read().decode(errors="ignore")

emails = content.split("\nFrom ")

for e in emails:
    try:
        msg = email.message_from_string(e)
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += str(part.get_payload())
        else:
            body = str(msg.get_payload())

        rows.append((body,0))
    except:
        pass

# ---------- FINAL DATASET ----------
df = pd.DataFrame(rows, columns=["text","label"])

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df.to_csv("final_phishing_dataset.csv",index=False)

print("Dataset created:",len(df),"emails")