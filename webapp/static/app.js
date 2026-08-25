(() => {
  const appRoot = document.body.dataset.appRoot || "";

  document.querySelectorAll("[data-nav-toggle]").forEach((button) => {
    const navigation = document.getElementById(button.getAttribute("aria-controls"));
    if (!navigation) return;
    button.addEventListener("click", () => {
      const open = navigation.classList.toggle("open");
      button.setAttribute("aria-expanded", String(open));
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    document.querySelectorAll("[data-site-nav].open").forEach((navigation) => {
      navigation.classList.remove("open");
      const button = document.querySelector(`[aria-controls="${navigation.id}"]`);
      button?.setAttribute("aria-expanded", "false");
    });
    document.querySelectorAll(".tools-menu[open]").forEach((details) => {
      details.removeAttribute("open");
    });
  });

  document.querySelectorAll("[data-file-zone]").forEach((zone) => {
    const input = zone.querySelector('input[type="file"]');
    const summary = zone.querySelector("[data-file-summary]");
    if (!input || !summary) return;

    const updateSummary = () => {
      const count = input.files?.length || 0;
      zone.classList.toggle("has-files", count > 0);
      if (!count) summary.textContent = "No files selected";
      else if (count === 1) summary.textContent = input.files[0].name;
      else summary.textContent = `${count} files selected`;
    };

    input.addEventListener("change", updateSummary);
    ["dragenter", "dragover"].forEach((name) => zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("has-files");
    }));
    ["dragleave", "drop"].forEach((name) => zone.addEventListener(name, (event) => {
      event.preventDefault();
      if (name === "drop" && event.dataTransfer?.files?.length) {
        input.files = event.dataTransfer.files;
        updateSummary();
      } else if (!input.files?.length) {
        zone.classList.remove("has-files");
      }
    }));
  });

  const presetValues = {
    docking: { n_conformers: "1", ph_min: "7.4", ph_max: "7.4", max_tautomers: "5" },
    enumerate: { n_conformers: "10", ph_min: "6.4", ph_max: "8.4", max_tautomers: "5" },
  };
  document.querySelectorAll("[data-preset]").forEach((preset) => {
    preset.addEventListener("change", () => {
      if (!preset.checked) return;
      Object.entries(presetValues[preset.value] || {}).forEach(([name, value]) => {
        const field = document.querySelector(`[name="${name}"]`);
        if (field) field.value = value;
      });
      const tautomers = document.querySelector('[name="tautomers"]');
      const ionise = document.querySelector('[name="ionise"]');
      const conformers = document.querySelector('[name="conformers"]');
      if (tautomers) tautomers.checked = preset.value === "enumerate";
      if (ionise) ionise.checked = true;
      if (conformers) conformers.checked = true;
    });
  });

  const pollCard = document.querySelector("[data-job-poll]");
  if (pollCard && pollCard.dataset.terminal !== "1") {
    const jobId = pollCard.dataset.jobPoll;
    const statusElement = document.querySelector("[data-job-status]");
    const stageElement = document.querySelector("[data-job-stage]");
    const messageElement = document.querySelector("[data-job-message]");
    const terminalStates = new Set(["succeeded", "failed", "canceled"]);

    const poll = async () => {
      try {
        const response = await fetch(`${appRoot}/api/jobs/${jobId}`, {
          credentials: "same-origin",
          headers: { Accept: "application/json" },
        });
        if (!response.ok) return;
        const job = await response.json();
        if (statusElement) {
          statusElement.className = `status-pill status-${job.status}`;
          const indicator = document.createElement("i");
          statusElement.replaceChildren(indicator, job.status.toUpperCase());
        }
        if (stageElement) stageElement.textContent = job.stage || "Queued";
        if (messageElement) messageElement.textContent = job.message || "Waiting for the worker";
        if (terminalStates.has(job.status)) window.location.reload();
      } catch (_) {
        // A temporary polling failure should not disrupt the job detail page.
      }
    };

    window.setInterval(poll, 5000);
  }
})();
