import React, { useRef, useState } from "react";

export default function HomePage() {
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUploadClick = () => fileInputRef.current.click();

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;

    setFile(f);
    setPreview(URL.createObjectURL(f));
    setReport(null);
    setError(null);
  };

  const handleAnalyse = async () => {
    if (!file) return alert("Upload an image first");

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("https://accessibility-backend.onrender.com/analyze"), {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Analysis failed");

      setReport(data);
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  };

  return (
    <div style={styles.page}>

      {/* HEADER */}
      <header style={styles.header}>
        <div style={styles.headerInner}>
          <img src="/white-logo.png" alt="" style={styles.logo} />
          <div style={styles.title}>Accessibility Audit AI</div>
        </div>
      </header>

      {/* MAIN */}
      <main style={styles.container}>

        <section style={styles.card}>
          <h2 style={styles.h2}>Analyse Infrastructure</h2>
          <p style={styles.subtext}>Upload an image in order to identify tactile flooring and assess it's compliance with DSAPT standards.</p>

          <div onClick={handleUploadClick} style={styles.uploadBox}>
            <input ref={fileInputRef} type="file" hidden onChange={handleFileChange} />

            {!preview ? (
              <div style={styles.uploadState}>Click to upload image</div>
            ) : (
              <img src={preview} alt="" style={styles.preview} />
            )}
          </div>

          <button onClick={handleAnalyse} disabled={loading} style={styles.button}>
            {loading ? "Analysing..." : "Run Analysis"}
          </button>

          {error && <div style={styles.error}>{error}</div>}
        </section>

        {report && (
          <section style={styles.card}>
            <h2 style={styles.h2}>Results</h2>
            <pre style={{ fontSize: 13 }}>{JSON.stringify(report, null, 2)}</pre>
          </section>
        )}

        <section style={styles.card}>
          <h2 style={styles.h2}>About this project</h2>
          <p style={styles.subtext}>
            This project was developed as part of a university initiative focused on applying AI to real-world accessibility challenges. It reflects a commitment to improving accessibility and supporting people with disabilities, with the broader aim of contributing to more inclusive public spaces. In this context, the project explores how technology can support greater independence and inclusion in everyday travel.
            <br/>
            <br/>
            The system is a proof-of-concept tool that uses computer vision to analyse public transport infrastructure and identify accessibility features and potential barriers. By processing images or video, it generates a structured, human-readable report to assist with accessibility assessment.
            <br/>
            <br/>
            As an early-stage prototype, the system has a number of limitations. Detection accuracy is influenced by factors such as image quality, lighting, and camera angles. In addition, some features may still require manual verification, and the tool is not intended to replace formal compliance assessments.
            
          </p>
        </section>

      </main>

      {/* FOOTER */}
      <footer style={styles.footer}>
        <div style={styles.footerInner}>
          <div style={styles.footerTitle}>About us</div>
          <div style={styles.footerText}>
          This project was developed by a small team of university students passionate about accessibility and inclusive design. Combining skills in AI, computer vision, and software development, the team set out to explore practical ways technology can improve everyday public transport experiences. Their goal is to create tools that support greater independence and accessibility for all users.          </div>
        </div>
      </footer>

    </div>
  );
}

/* ---------------- STYLES ---------------- */

const styles = {
  page: {
    fontFamily: "Arial",
    background: "#f8fafc",
    overflowX: "hidden",
  },

  /* DARK GREY HEADER */
  header: {
    background: "#212121",
    color: "#fff",
    borderBottom: "1px solid #111827",
  },

  headerInner: {
    maxWidth: 1100,
    margin: "0 auto",
    display: "grid",
    gridTemplateColumns: "1fr auto 1fr",
    alignItems: "center",
    padding: "14px 16px",
  },

  logo: {
    height: 28,
    objectFit: "contain",
  },

  title: {
    textAlign: "center",
    fontWeight: 700,
    fontSize: 18,
    color: "#ffffff",
  },

  container: {
    maxWidth: 1100,
    margin: "0 auto",
    padding: 20,
    display: "flex",
    flexDirection: "column",
    gap: 18,
  },

  card: {
    background: "#fff",
    border: "1px solid #e5e7eb",
    borderRadius: 16,
    padding: 20,
  },

  h2: {
    fontSize: 17,
    fontWeight: 600,
    marginBottom: 10,
  },

  subtext: {
    fontSize: 14.5,
    color: "#4b5563",
    lineHeight: 1.6,
  },

  uploadBox: {
    height: 260,
    background: "#f1f5f9",
    border: "1px dashed #cbd5e1",
    borderRadius: 14,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    overflow: "hidden",
  },

  uploadState: { color: "#64748b" },

  preview: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },

  button: {
    marginTop: 12,
    width: "100%",
    padding: 11,
    background: "#2563eb",
    color: "#fff",
    border: 0,
    borderRadius: 10,
    fontWeight: 600,
    cursor: "pointer",
  },

  error: {
    marginTop: 10,
    background: "#fee2e2",
    color: "#b91c1c",
    padding: 10,
    borderRadius: 10,
  },

  /* DARK GREY FOOTER */
  footer: {
    marginTop: 30,
    background: "#212121",
    color: "#e5e7eb",
    padding: "28px 16px",
  },

  footerInner: {
    maxWidth: 1100,
    margin: "0 auto",
  },

  footerTitle: {
    fontWeight: 700,
    marginBottom: 10,
    color: "#ffffff",
  },

  footerText: {
    fontSize: 14.5,
    lineHeight: 1.7,
    color: "#d1d5db",
  },
};
