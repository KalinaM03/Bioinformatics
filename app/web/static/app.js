(() => {
  function escapeHtml(s){
    return String(s)
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;")
      .replaceAll('"',"&quot;")
      .replaceAll("'","&#039;");
  }

  async function postJson(url, payload, msgSel){
    const msg = document.querySelector(msgSel);
    if (msg) { msg.textContent = "Записвам…"; msg.classList.remove("danger"); }
    try{
      const res = await fetch(url, { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify(payload)});
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || ("HTTP " + res.status));
      }
      if (msg) { msg.textContent = "Готово ✅"; }
    }catch(e){
      if (msg) { msg.textContent = "Грешка: " + (e.message || e); msg.classList.add("danger"); }
    }
  }

  const jobsTable = document.querySelector("[data-jobs-table]");
  if (jobsTable) {
    const statusEl = document.querySelector("[data-jobs-status]");
    async function refreshJobs(){
      try{
        const res = await fetch("/api/jobs?limit=50");
        const jobs = await res.json();
        const tbody = jobsTable.querySelector("tbody");
        tbody.innerHTML = "";
        for (const j of jobs){
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td>${escapeHtml(j.created_at_utc || "")}</td>
            <td>${escapeHtml(j.folder || "")}</td>
            <td><span class="badge">${escapeHtml(j.hash_mode || "")}</span></td>
            <td>
              <span class="badge">${escapeHtml(j.status || "")}</span>
              ${j.error_message ? `<div class="muted danger">${escapeHtml(j.error_message)}</div>` : ""}
            </td>
            <td class="right">${j.files_seen ?? 0}</td>
            <td class="right">${j.files_indexed ?? 0}</td>
            <td class="right">${j.files_updated ?? 0}</td>
            <td class="right">${j.files_failed ?? 0}</td>
          `;
          tbody.appendChild(tr);
        }
        if (statusEl) statusEl.textContent = "Обновено: " + new Date().toLocaleTimeString();
      } catch(e){
        if (statusEl) statusEl.textContent = "Грешка при обновяване.";
      }
    }
    refreshJobs();
    setInterval(refreshJobs, 2500);
  }

  const fieldsForm = document.querySelector("[data-fields-form]");
  if (fieldsForm) {
    fieldsForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const fileId = fieldsForm.getAttribute("data-file-id");
      const payload = {
        project: (fieldsForm.querySelector("[name=project]").value || "").trim() || null,
        sample_id: (fieldsForm.querySelector("[name=sample_id]").value || "").trim() || null,
        patient_pseudo_id: (fieldsForm.querySelector("[name=patient_pseudo_id]").value || "").trim() || null,
      };
      await postJson(`/api/files/${fileId}/fields`, payload, "[data-fields-msg]");
    });
  }

  const tagsForm = document.querySelector("[data-tags-form]");
  if (tagsForm) {
    tagsForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const fileId = tagsForm.getAttribute("data-file-id");
      const raw = (tagsForm.querySelector("[name=tags]").value || "");
      const tags = raw.split(",").map(s => s.trim()).filter(Boolean);
      await postJson(`/api/files/${fileId}/tags`, tags, "[data-tags-msg]");
      setTimeout(() => location.reload(), 600);
    });
  }
})();
