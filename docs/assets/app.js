/* Static documentation enhancements. The content remains readable without JavaScript. */
(() => {
  const upstream = 'https://github.com/xArm-Developer/lerobot_robot_ufactory/blob/5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82/';
  const fork = 'https://github.com/George121380/lerobot_robot_ufactory/blob/main/';
  document.querySelectorAll('[data-source]').forEach(a => { a.href = upstream + a.dataset.source; });
  document.querySelectorAll('[data-repo]').forEach(a => { a.href = fork + a.dataset.repo; });

  const root = document.documentElement;
  const themeButton = document.querySelector('#theme-toggle');
  let preference;
  try { preference = localStorage.getItem('xarm-guide-theme'); } catch (_) { /* storage may be disabled */ }
  let dark = preference ? preference === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  const updateTheme = () => {
    root.dataset.theme = dark ? 'dark' : 'light';
    themeButton.textContent = dark ? 'Light' : 'Dark';
    themeButton.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  };
  updateTheme();
  themeButton.addEventListener('click', () => {
    dark = !dark; updateTheme();
    try { localStorage.setItem('xarm-guide-theme', dark ? 'dark' : 'light'); } catch (_) { /* optional preference */ }
  });

  const menuButton = document.querySelector('#menu-toggle');
  const closeMenu = () => { document.body.classList.remove('menu-open'); menuButton.setAttribute('aria-expanded', 'false'); };
  menuButton.addEventListener('click', () => {
    const open = document.body.classList.toggle('menu-open');
    menuButton.setAttribute('aria-expanded', String(open));
  });
  document.querySelectorAll('#sidebar a').forEach(a => a.addEventListener('click', closeMenu));
  document.querySelector('main').addEventListener('click', closeMenu);

  const sections = [...document.querySelectorAll('.doc-section')];
  const navLinks = [...document.querySelectorAll('#navigation a')];
  let framePending = false;
  const setActive = () => {
    const section = [...sections].reverse().find(s => s.getBoundingClientRect().top <= 160) || sections[0];
    navLinks.forEach(a => {
      const active = a.hash === '#' + section.id;
      a.classList.toggle('active', active);
      if (active) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current');
    });
    framePending = false;
  };
  window.addEventListener('scroll', () => { if (!framePending) { framePending = true; requestAnimationFrame(setActive); } }, { passive: true });
  setActive();

  const search = document.querySelector('#search');
  const results = document.querySelector('#search-results');
  const navigation = document.querySelector('#navigation');
  const index = sections.map(s => {
    const walker = document.createTreeWalker(s, NodeFilter.SHOW_TEXT);
    const text = [];
    while (walker.nextNode()) text.push(walker.currentNode.nodeValue);
    return { id: s.id, title: s.dataset.title, text: text.join(' ').replace(/\s+/g, ' ').trim() };
  });
  search.addEventListener('input', () => {
    const query = search.value.trim().toLowerCase();
    results.replaceChildren();
    results.hidden = !query;
    navigation.hidden = !!query;
    if (!query) return;
    const terms = query.split(/\s+/);
    const matches = index.filter(s => terms.every(t => s.text.toLowerCase().includes(t)));
    const label = document.createElement('p');
    label.textContent = `${matches.length} matching section${matches.length === 1 ? '' : 's'}`;
    results.append(label);
    matches.forEach(s => {
      const a = document.createElement('a'); a.href = '#' + s.id;
      const title = document.createElement('strong'); title.textContent = s.title;
      const excerpt = document.createElement('span');
      const at = Math.max(0, s.text.toLowerCase().indexOf(terms[0]) - 35);
      excerpt.textContent = (at ? '…' : '') + s.text.slice(at, at + 125) + '…';
      a.append(title, excerpt); a.addEventListener('click', closeMenu); results.append(a);
    });
  });
  document.addEventListener('keydown', event => {
    const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '') || document.activeElement?.isContentEditable;
    if (event.key === '/' && !typing && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault();
      if (window.matchMedia('(max-width: 800px)').matches) { document.body.classList.add('menu-open'); menuButton.setAttribute('aria-expanded', 'true'); }
      search.focus();
    }
    if (event.key === 'Escape') {
      if (search.value) { search.value = ''; search.dispatchEvent(new Event('input')); }
      closeMenu();
    }
  });

  const toast = document.querySelector('#toast');
  let toastTimer;
  document.querySelectorAll('pre').forEach(pre => {
    const wrapper = document.createElement('div'); wrapper.className = 'code-wrapper';
    pre.before(wrapper); wrapper.append(pre);
    const button = document.createElement('button'); button.className = 'copy-button'; button.type = 'button';
    button.textContent = 'Copy'; button.setAttribute('aria-label', 'Copy code block');
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector('code').textContent);
        button.textContent = 'Copied'; toast.textContent = 'Command copied';
      } catch (_) {
        button.textContent = 'Select text'; toast.textContent = 'Clipboard unavailable. Select and copy the code manually.';
      }
      toast.hidden = false; clearTimeout(toastTimer);
      toastTimer = setTimeout(() => { toast.hidden = true; button.textContent = 'Copy'; }, 2400);
    });
    wrapper.append(button);
  });
})();
