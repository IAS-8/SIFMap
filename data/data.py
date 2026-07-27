# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from data.utils import get_checksum
import pickle as pkl
import os


class Datastruct(object):
    """
    This abstract class defines a data structure that holds image references (via a object of type _ImageData),
     a matching matrix and image correspondences. It serves as an interface with the registration and alignment
     implementations defined under `match` and `align`, ensuring safe removal of images and correspondences.

    """
    def __init__(self, image_data, protect_correspondences_in_close_images=0, close_images_limit=1, 
                 from_file=None, spread_points=True, **kwargs):
        """

        Args:
            image_data: _ImageData object
            protect_correspondences_in_close_images: whether to protect close images (i.e. images with a difference in
                                                     numerical index smaller than close_images_limit. For this to be
                                                     taken account of the image_data object must hold a field `idnum`.
            close_images_limit: threshold to consider two images close
            from_file: whether to load a preexisting state from disk
            spread_points: whether to filter correspondences in a way to maximize distribution across individual images.
        """
        self.data = image_data
        self.matches = None
        n_images = len(self.data)

        self._mapping = np.arange(n_images).astype(int)
        self.protect_correspondences_in_close_images = protect_correspondences_in_close_images
        self.close_images_limit = close_images_limit
        self.spread_points = spread_points

        if from_file is None:
            self.match_matrix = np.empty((n_images, n_images), dtype=object)
            self._mapping = np.arange(n_images).astype(int)
            self.load_indices = None

        else:
            with open(from_file, 'rb') as fil:
                match_matrix, _ids, _ = pkl.load(fil)

            dels = [i for i, imname in enumerate(self.ids) if imname not in _ids]
            perm = [_ids.index(imname) for imname in self.ids if imname in _ids]
            
            self._mapping = np.delete(self._mapping, dels)

            self.match_matrix = match_matrix[perm, :][:, perm]
            self.matches = self._get_matches_from_matchmat()
            self.load_indices = perm

            #assert checksum == get_checksum(self.data._data)

    def __len__(self):
        return len(self._mapping)

    @property
    def images(self):
        return [self.data.data[i] for i in self._mapping]

    @property
    def paths(self):
        return [self.data.paths[i] for i in self._mapping]

    @property
    def ids(self):
        return [self.data.ids[i] for i in self._mapping]

    @property
    def mapping(self):
        return self._mapping

    @property
    def shapes(self):
        shapes = self.data.shapes
        return [shapes[i] for i in self._mapping]

    def to_file(self, to_file):

        with open(to_file, 'wb') as fil:
            save_data = (self.match_matrix, self.ids, get_checksum(self.data._data))
            pkl.dump(save_data, fil)

    def correspondence_statistics(self):
        return dict(tot_n_corr=self.matches[:, 2].sum(),
                    min_n_corr = self.matches[:, 2].min(),
                    max_n_corr=self.matches[:, 2].max(),
                    median_n_corr=np.median(self.matches[:, 2]),
                    n_pairs=self.matches[:, 2].shape[0],
                    )

    def remove_isolated_images(self, min_n_matches=1):
        """
            This function checks the overlapping image pairs for each image
            and remove the ones that have less than min_n_matches image pairs.

        Args:
            min_n_matches ():

        Returns:
            remove : Image indexes to be removed or marked as isolated.
        """
        matches = self.matches
        n_images = len(self._mapping)
        
        matches = np.concatenate((matches[:, 0], matches[:, 1]))
        remove = []
        for r in np.arange(n_images):
            mat = np.sum(matches == r)
            if mat < min_n_matches:
                remove.append(r)

        remove = np.array(remove).astype(int)

        self.remove(remove)
        return remove

    def remove_small_movements(self, min_movement=1, n_points=1):
        """

        """
        matches = self.matches
        n_images = len(self._mapping)

        remove = []
        for i in range(len(matches)):
            Im1 = int(matches[i][0])
            Im2 = int(matches[i][1])

            Im2PList = self.match_matrix[Im1][Im2]
            Im1PList = self.match_matrix[Im2][Im1]

            if len(Im2PList) == 0 or len(Im1PList) == 0:
                continue
            
            distances = np.array([np.sqrt(((p1 - p2) ** 2).sum()) for p1, p2, in zip(Im1PList, Im2PList)])
            if np.quantile(distances, 0.75) < min_movement:
                n_keeps = min(n_points, len(Im1PList))
                keep_ids = spread_points(np.concatenate((Im1PList.T, Im2PList.T)), n_keeps)
                all_ids = list(range(len(Im2PList)))
                del_ids = set(all_ids) - set(keep_ids)
                if len(del_ids) > 0:
                    remove.append((Im1, Im2, np.asarray(list(del_ids))))
        
        if len(remove) > 0:
            n_removals = np.sum([len(del_ids) for (im1, im2, del_ids) in remove])
            print(f'Removing {n_removals} correspondences due to too small movement.')

        for rem in remove:
            self.remove_correspondence(*rem[:2], correspondence_id=rem[2])

    def remove_pairs(self, pairs):
        for tt in range(pairs.shape[0]):
            self.match_matrix[pairs[tt, 0], pairs[tt, 1]] = None
            self.match_matrix[pairs[tt, 1], pairs[tt, 0]] = None

    def is_close_images_and_protected(self, Im1, Im2, to_be_removed=0):
        if not ('idnum' in self.data.data[self._mapping[Im1]] and 'idnum' in self.data.data[self._mapping[Im2]]):
            return False

        if self.protect_correspondences_in_close_images > 0 \
                and np.abs(self.data.data[self._mapping[Im1]]['idnum'] - self.data.data[self._mapping[Im2]]['idnum']) <= self.close_images_limit \
                and len(self.match_matrix[Im1, Im2]) - to_be_removed <= self.protect_correspondences_in_close_images:
            return True

        else:
            return False

    def is_close_images(self, Im1, Im2):
        if not ('idnum' in self.data.data[self._mapping[Im1]] and 'idnum' in self.data.data[self._mapping[Im2]]):
            return False

        if self.protect_correspondences_in_close_images > 0 \
                and np.abs(self.data.data[self._mapping[Im1]]['idnum'] - self.data.data[self._mapping[Im2]]['idnum']) <= self.close_images_limit:
            return True

        else:
            return False


    def remove_correspondence(self, Im1, Im2, correspondence_coord=None, correspondence_id=None):
        """
        This function is to remove a single correspondences between an image pair.
        Args:
            correspondence_coord (): correspondences data to be removed
            Im1 (): overlapping image pair index of the first image
            Im2 (): overlapping image pair index of the second image

        Returns:
        """
       
        # Remove correspondences 'pm' from the correspondences list between 'Im1' and 'Im2'
        if correspondence_coord is not None:
            delID1 = np.where(np.abs(P[:, 0] - correspondence_coord[0] + P[:, 1] - correspondence_coord[1]) < 1e-2)[0]
            delID2 = np.where(np.abs(M[:, 0] - correspondence_coord[2] + M[:, 1] - correspondence_coord[3]) < 1e-2)[0]
            delID = np.intersect1d(delID1, delID2)

        else:
            delID = correspondence_id
        
        # check if images are close
        # if close only remove if there are more than n_close_correspondences
        if self.is_close_images_and_protected(Im1, Im2, to_be_removed=len(delID)):
            return

        M = np.array(self.match_matrix[Im1, Im2])
        P = np.array(self.match_matrix[Im2, Im1])


        P = np.delete(P, delID, axis=0)
        M = np.delete(M, delID, axis=0)

        self.match_matrix[Im1, Im2] = M#.tolist()
        self.match_matrix[Im2, Im1] = P#.tolist()

    def remove(self, remove):
        """
            This function removes images from the dataset via its index number
        Args:
            data (): Data structure to keep correspondences data
                among successfully matched image pairs
            remove (): image indices to be removed

        Returns:
            new_matching: a new Data structure with images removed.
        """
        if 'idnum' in self.data.data[0]:
            print('Removing images (image IDs): ', [self.data.data[self._mapping[r]]['idnum'] for r in remove])

        else:
            print('Removing images: ', remove)
        
        self.match_matrix = np.delete(self.match_matrix, remove, axis=0)
        self.match_matrix = np.delete(self.match_matrix, remove, axis=1)

        self._mapping = np.delete(self._mapping, remove)

        # Delete in MATCHES
        _, b1, _ = np.intersect1d(self.matches[:, 0], remove, return_indices=True)
        _, b2, _ = np.intersect1d(self.matches[:, 1], remove, return_indices=True)
        b = np.unique(np.concatenate((b1, b2)))

        self.matches = np.delete(self.matches, b, axis=0)

        self.matches[:, 0] = self._on_remove_adapt_indices(np.asarray(self.matches)[:, 0], remove)
        self.matches[:, 1] = self._on_remove_adapt_indices(np.asarray(self.matches)[:, 1], remove)

    def _on_remove_adapt_indices(self, indices, remove):
        remove = np.asarray(sorted(remove))
        for i in range(len(remove)):
            _mask = np.where(indices > remove[i])
            indices[_mask] -= 1
            remove -= 1

        return indices

    def _get_matches_from_matchmat(self, min_n_correspondences=None):
        """
            This function converts the overlapping image pairs data in matrix
            to a list
        Args:
            min_n_correspondences (): a threshold on total number of correspondences to be counted as
                successfully matched

        Returns:
            out : a list of overlapping image pairs
        """

        if min_n_correspondences is None:
            min_n_correspondences = 0

        n_images = self.match_matrix.shape[0]
        out = list()
        for i in range(n_images - 1):
            for j in range(i + 1, n_images):
                if self.match_matrix[i, j] is None:
                    continue
                
                if not self.is_close_images(i, j):
                    enough_correspondences = len(self.match_matrix[i, j]) >= min_n_correspondences 
                else:
                    enough_correspondences = len(self.match_matrix[i, j]) > 0

                if enough_correspondences:
                    _match = [j, i, len(self.match_matrix[i, j])]
                    out.append(_match)

        out = np.array(out).astype(int)
        return out

    def _set_matches(self, min_n_correspondences=None, n_points=None):
        """
            This function (i) removes isolated images and (ii) returns matches with more correspondences
            than min_n_correspondences.

        Args:
            data : data structure keeping the correspondences data
            min_n_correspondences : a minimum number of correspondences for an image pair.

        Returns:
            data : data structure keeping the correspondences data
        """

        # remove matches with too few correspondences
        if min_n_correspondences is not None:
            self.matches = self._get_matches_from_matchmat(min_n_correspondences)

        if n_points is not None:
            matches = self.matches
            for i in range(len(matches)):
                Im1 = int(matches[i][0])
                Im2 = int(matches[i][1])
                n_matching_points = self.match_matrix[Im1][Im2].shape[0]

                if n_matching_points == 0:
                    continue

                Im2PList = self.match_matrix[Im1][Im2].T
                Im1PList = self.match_matrix[Im2][Im1].T

                if n_matching_points > n_points:
                    if self.spread_points:
                        idd = spread_points(np.concatenate((Im1PList, Im2PList)), n_points)
                        self.match_matrix[Im1][Im2] = Im2PList[:, idd].T
                        self.match_matrix[Im2][Im1] = Im1PList[:, idd].T

                    else:
                        self.match_matrix[Im1][Im2] = Im2PList[:, :n_points].T
                        self.match_matrix[Im2][Im1] = Im1PList[:, :n_points].T

                    self.matches[i, 2] = n_points

        return self.matches

    def update_matches(self, min_n_correspondences=None, n_points=None,
                       min_movement=None):
        """
            This function updates the final list of overlapping image pairs
        Args:
            data (): Data structure to keep correspondences data
            among successfully matched image pairs
            min_n_correspondences (): a minimum number of correspondences for an image
            pair to be considered as successfully matched
            n_points (): a number of correspondences to be used during global alignment

        Returns:
            remove : a list of images that are removed as they do not have
            overlapping pairs (less than 2) within the dataset.

        """
        if n_points is not None and min_n_correspondences is not None:
            assert n_points >= min_n_correspondences

        if min_movement is not None:
            self.remove_small_movements(min_movement=min_movement, 
                                        n_points=max(int(min_n_correspondences * 1.1), 6))

        self._set_matches(min_n_correspondences=min_n_correspondences, n_points=n_points)
        remove = self.remove_isolated_images()

        return remove


class _ImageData(object):
    """
    This is an abstract class interfacing a set of images with the Datastruct class.

    """
    BASE_KEYS = ['points', 'features', 'valid_points', 'file_name']

    def setup(self, paths, ids):
        """
        Args:
            n_images ():

        Returns:

        """
        data = np.empty(len(paths), dtype=object)
        for i in range(len(paths)):
            data[i] = dict([(b, None) for b in self.BASE_KEYS])
            data[i].update(dict(id=ids[i]))

            try:
                data[i].update(idnum=int(ids[i]))
            except ValueError as e:
                pass

        return data

    def __len__(self):
        return None

    @property
    def shapes(self):
        return None

    @property
    def data(self):
        return None

    @property
    def paths(self):
        return None


def spread_points(point_matches, NumCorresp):
    """
        This function selects correspondences distributed among image
    Args:
        point_matches (): correspondences locations
        NumCorresp (): number of correspondences to be selected.

    Returns:
        I : indices of selected correspondences
    """
    # Number of Nodes
    PM = point_matches.copy()
    l, c = PM.shape

    if l < 4 or c < NumCorresp:
        raise ValueError('Input PM matrix must be a 4xc matrix (c>=NumCorresp)')

    if NumCorresp == c:
        I = np.arange(NumCorresp)  # + 1
    else:
        XMin = np.min(PM[0, :])
        XMax = np.max(PM[0, :])
        YMin = np.min(PM[1, :])
        YMax = np.max(PM[1, :])
        XCen = (XMin + XMax) / 2
        YCen = (YMin + YMax) / 2
        XMean = np.mean(PM[0, :])
        YMean = np.mean(PM[1, :])

        InterestPoints = np.array([[XMin, YMin],
                                   [XMax, YMax],
                                   [XMin, YMax],
                                   [XMax, YMin],
                                   [XMean, YMean],
                                   [XMin, YCen],
                                   [XMax, YCen],
                                   [XCen, YMin],
                                   [XCen, YMax],
                                   [XCen, YCen]])

        NumInterestPoints = min(NumCorresp, 9)
        DistPoints = np.zeros((NumInterestPoints, c))

        for i in range(NumInterestPoints):
            Points = PM[:2, :]
            Points[0, :] -= InterestPoints[i, 0]
            Points[1, :] -= InterestPoints[i, 1]
            Points = Points * Points
            Points = np.sqrt(np.sum(Points, axis=0))
            DistPoints[i, :] = Points

        I = np.zeros(NumCorresp, dtype=int)

        for i in range(NumCorresp):
            j = (i - 1) % NumInterestPoints
            Idx = np.argmin(DistPoints[j, :])
            DistPoints[:, Idx] = np.inf
            I[i] = Idx

    return I
