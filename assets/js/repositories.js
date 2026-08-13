document.addEventListener("DOMContentLoaded", function () {
  const userCards = document.querySelectorAll("[data-github-user]");
  const repoCards = document.querySelectorAll("[data-github-repo]");

  function formatDate(value) {
    try {
      return new Date(value).toLocaleDateString("en", { year: "numeric", month: "short", day: "numeric" });
    } catch (e) {
      return value || "";
    }
  }

  function renderUserCard(card, data) {
    const blog = data.blog
      ? `<a class="repo-inline-link" href="${data.blog}" target="_blank" rel="noopener noreferrer">${data.blog.replace(/^https?:\/\//, "")}</a>`
      : "";
    const location = data.location ? `<span>${data.location}</span>` : "";
    card.classList.remove("repo-skeleton");
    card.innerHTML = `
      <a class="repo-card-link" href="${data.html_url}" target="_blank" rel="noopener noreferrer">
        <div class="repo-user-top">
          <img class="repo-user-avatar-img" src="${data.avatar_url}" alt="${data.login}">
          <div class="repo-user-meta">
            <h3>${data.name || data.login}</h3>
            <p>@${data.login}</p>
          </div>
        </div>
        <p class="repo-user-bio">${data.bio || "GitHub profile"}</p>
        <div class="repo-user-foot">
          <div class="repo-stat-row">
            <span class="repo-stat-pill"><strong>${data.public_repos}</strong> repos</span>
            <span class="repo-stat-pill"><strong>${data.followers}</strong> followers</span>
            <span class="repo-stat-pill"><strong>${data.following}</strong> following</span>
          </div>
          <div class="repo-inline-meta">${location}${blog ? `<span>·</span>${blog}` : ""}</div>
        </div>
      </a>
    `;
  }

  function renderRepoCard(card, data) {
    const topics = (data.topics || [])
      .slice(0, 3)
      .map((topic) => `<span class="repo-topic">${topic}</span>`)
      .join("");
    card.classList.remove("repo-skeleton");
    card.innerHTML = `
      <a class="repo-card-link" href="${data.html_url}" target="_blank" rel="noopener noreferrer">
        <div class="repo-card-head">
          <span class="repo-owner">${data.owner.login}</span>
          <span class="repo-chip">${data.visibility || "public"}</span>
        </div>
        <h3>${data.name}</h3>
        <p class="repo-description">${data.description || "No description provided."}</p>
        <div class="repo-topic-row">${topics}</div>
        <div class="repo-stat-row">
          <span class="repo-stat-pill"><i class="fa-solid fa-star fa-sm"></i> ${data.stargazers_count}</span>
          <span class="repo-stat-pill"><i class="fa-solid fa-code-fork fa-sm"></i> ${data.forks_count}</span>
          <span class="repo-stat-pill">${data.language || "Code"}</span>
        </div>
        <div class="repo-inline-meta">
          <span>Updated ${formatDate(data.updated_at)}</span>
          ${data.license && data.license.name ? `<span>·</span><span>${data.license.name}</span>` : ""}
        </div>
      </a>
    `;
  }

  function renderFallback(card, title, url, description) {
    card.classList.remove("repo-skeleton");
    card.innerHTML = `
      <a class="repo-card-link" href="${url}" target="_blank" rel="noopener noreferrer">
        <h3>${title}</h3>
        <p class="repo-description">${description}</p>
      </a>
    `;
  }

  userCards.forEach(function (card) {
    const username = card.dataset.githubUser;
    fetch(`https://api.github.com/users/${username}`)
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error("Failed to load user"))))
      .then((data) => renderUserCard(card, data))
      .catch(() => renderFallback(card, username, `https://github.com/${username}`, "Open GitHub profile"));
  });

  repoCards.forEach(function (card) {
    const repo = card.dataset.githubRepo;
    fetch(`https://api.github.com/repos/${repo}`)
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error("Failed to load repo"))))
      .then((data) => renderRepoCard(card, data))
      .catch(() => renderFallback(card, repo, `https://github.com/${repo}`, "Open repository on GitHub"));
  });
});
