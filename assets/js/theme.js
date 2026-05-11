(function(){
  var root=document.documentElement;
  var btn=document.getElementById('themeToggle');
  if(!btn) return;
  var modes=['auto','light','dark'];

  function readTheme(){
    try{
      var ls=localStorage.getItem('theme');
      if(ls) return ls;
    }catch(e){}
    var m=document.cookie.match(/(?:^|;\s*)theme=([^;]+)/);
    return m?decodeURIComponent(m[1]):'auto';
  }
  function writeTheme(v){
    try{localStorage.setItem('theme',v);}catch(e){}
    // 1-year cookie; path=/ so it's shared across the site
    var c='theme='+encodeURIComponent(v)+';path=/;max-age=31536000';
    if(location.protocol!=='file:') c+=';samesite=lax';
    try{document.cookie=c;}catch(e){}
  }

  apply(readTheme());

  btn.addEventListener('click',function(){
    var cur=readTheme();
    var next=modes[(modes.indexOf(cur)+1)%modes.length];
    if(modes.indexOf(cur)<0) next='light';
    writeTheme(next);
    apply(next);
  });

  function apply(mode){
    if(mode==='auto'){root.removeAttribute('data-theme')}
    else{root.setAttribute('data-theme',mode)}
    btn.textContent=mode;
  }
})();
