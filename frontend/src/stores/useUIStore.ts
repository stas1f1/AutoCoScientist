import { create } from 'zustand'

interface UIState {
  sidebarOpen: boolean
  sidebarCollapsed: boolean
  fileExplorerOpen: boolean
  datasetManagerOpen: boolean
  libraryInstallerOpen: boolean
  filePreviewPath: string | null

  // Actions
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  setSidebarCollapsed: (collapsed: boolean) => void
  openFileExplorer: () => void
  closeFileExplorer: () => void
  openDatasetManager: () => void
  closeDatasetManager: () => void
  openLibraryInstaller: () => void
  closeLibraryInstaller: () => void
  setFilePreviewPath: (path: string | null) => void
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  sidebarCollapsed: false,
  fileExplorerOpen: false,
  datasetManagerOpen: false,
  libraryInstallerOpen: false,
  filePreviewPath: null,

  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
  openFileExplorer: () => set({ fileExplorerOpen: true }),
  closeFileExplorer: () => set({ fileExplorerOpen: false, filePreviewPath: null }),
  openDatasetManager: () => set({ datasetManagerOpen: true }),
  closeDatasetManager: () => set({ datasetManagerOpen: false }),
  openLibraryInstaller: () => set({ libraryInstallerOpen: true }),
  closeLibraryInstaller: () => set({ libraryInstallerOpen: false }),
  setFilePreviewPath: (path) => set({ filePreviewPath: path }),
}))
