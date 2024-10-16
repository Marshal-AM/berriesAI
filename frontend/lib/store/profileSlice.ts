import { StateCreator } from "zustand";

interface ProfileState {
  username: string;
  name: string;
  bio: string;
  image: string;
  following: number;
  followers: number;
  niches: string[];
  preferences: string[];
}

interface ProfileActions {
  update: (profile: ProfileState) => void;
}

export type ProfileSlice = ProfileState & ProfileActions;

export const initalProfileState: ProfileState = {
  username: "",
  name: "",
  bio: "",
  image: "",
  following: 0,
  followers: 0,
  niches: [],
  preferences: [],
};

export const createProfileSlice: StateCreator<
  ProfileSlice,
  [],
  [],
  ProfileSlice
> = (set) => ({
  ...initalProfileState,
  update: (profile) => set((state) => ({ ...state, ...profile })),
});
