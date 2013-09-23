.onLoad<-function(libname, pkgname){
	e<-emptyenv()
	library.dynam('robgpu', pkgname, libname)
	reg.finalizer(e,function(...){unloadNamespace(pkgname)}, onexit=T)
}

.onUnload<-function(libpath){
	library.dynam.unload('robgpu', libpath)
}
