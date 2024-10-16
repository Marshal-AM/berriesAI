import { createStore } from "zustand";
import { createProfileSlice, ProfileSlice } from "./profileSlice";
export type EnvironmentStore = ProfileSlice;

export const createEnvironmentStore = () =>
  createStore<EnvironmentStore>()((...a) => ({
    ...createProfileSlice(...a),
  }));
