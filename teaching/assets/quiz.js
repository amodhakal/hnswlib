// Minimal quiz widget — reusable across lessons
// Usage: <div class="quiz" data-answer="B"></div> with buttons data-choice="A"...
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.quiz').forEach(q => {
    const answer = q.dataset.answer;
    const fb = q.querySelector('.feedback');
    q.querySelectorAll('button[data-choice]').forEach(btn => {
      btn.addEventListener('click', () => {
        q.querySelectorAll('button').forEach(b=>b.classList.remove('correct','wrong'));
        if (btn.dataset.choice === answer) { btn.classList.add('correct'); if(fb) fb.textContent='✓ Correct — '+ (fb.dataset.ok||''); }
        else { btn.classList.add('wrong'); if(fb) fb.textContent='✗ Try again — '+ (fb.dataset.no||''); }
      });
    });
  });
});
