document.getElementById("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById("imageInput");
  if (!fileInput.files.length) {
    alert("Please select an image file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Server responded with status ${res.status}`);
    }

    const data = await res.json();
    document.getElementById("result").innerText = `Prediction: ${data.result}`;
  } catch (error) {
    document.getElementById("result").innerText = `Error: ${error.message}`;
  }
});
