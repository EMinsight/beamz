const sceneSpec = window.__ZVIEW_SCENE__;
document.title = sceneSpec.title || "ZView";

mountZView({
  el: document.getElementById("zview-root"),
  sceneSpec,
});
