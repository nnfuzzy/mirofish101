/**
 * 临时存储待上传的文件和需求
 * 用于首页点击启动引擎后立即跳转，在Process页面再进行API调用
 *
 * Files cannot be serialized to sessionStorage (File objects lose their
 * binary contents on JSON round-trip), so files stay in-memory only.
 * The simulationRequirement text is persisted so users don't lose their
 * prompt on accidental reload.
 */
import { reactive, watch } from 'vue'

const STORAGE_KEY = 'mirofish.pendingUpload.requirement'

const restoredRequirement =
  typeof sessionStorage !== 'undefined'
    ? sessionStorage.getItem(STORAGE_KEY) || ''
    : ''

const state = reactive({
  files: [],
  simulationRequirement: restoredRequirement,
  isPending: false
})

if (typeof sessionStorage !== 'undefined') {
  watch(
    () => state.simulationRequirement,
    (val) => {
      if (val) sessionStorage.setItem(STORAGE_KEY, val)
      else sessionStorage.removeItem(STORAGE_KEY)
    }
  )
}

export function setPendingUpload(files, requirement) {
  state.files = files
  state.simulationRequirement = requirement
  state.isPending = true
}

export function getPendingUpload() {
  return {
    files: state.files,
    simulationRequirement: state.simulationRequirement,
    isPending: state.isPending
  }
}

export function clearPendingUpload() {
  state.files = []
  state.simulationRequirement = ''
  state.isPending = false
}

export default state
